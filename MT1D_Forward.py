# MT1D_forward_smart_full.py
import numpy as np
import math
import time
import multiprocessing
from joblib import Parallel, delayed

mu0 = 4.0 * math.pi * 1e-7
epsilon0 = 8.854187817e-12
tol = 1e-14


# Globals used to reduce pickling overhead in parallel workers
_EP_GLOBAL = None
_N_GLOBAL = None
_FREQS_GLOBAL = None

def _mt_worker(freq):
    """Worker for Parallel: uses global ep & n"""
    return MT1D(_N_GLOBAL, _EP_GLOBAL, float(freq))

def MT1D(n, ep, f):
    """
    单频完整递推
    输入:
      n: 层数 (int)
      ep: 层参数数组 shape (n,17) 按照你的原始格式
      f: 频率 (Hz)
    返回:
      App (2x2 float), Phase (2x2 float), Zsurf (2x2 complex)
    """
    omega = 2.0 * math.pi * f

    # allocate
    epsilon = np.zeros((n+1, 3, 3), dtype=np.complex128)
    A = np.zeros((n+1, 2, 2), dtype=np.complex128)
    y_hat = np.zeros((n+1, 3, 3), dtype=np.complex128)
    z_hat = np.zeros(n+1, dtype=np.complex128)
    depth = np.zeros(n+1, dtype=np.float64)
    h = np.zeros(n+1, dtype=np.float64)
    sigma = np.zeros((n+1, 3, 3), dtype=np.complex128)

    # 1) init per-layer
    for i in range(1, n+1):
        alfa = ep[i-1, 3] * math.pi / 180.0
        beta = ep[i-1, 4] * math.pi / 180.0
        gama = ep[i-1, 5] * math.pi / 180.0

        # main conductivities (ep columns 0..2 are sigma_xx,sigma_yy,sigma_zz)
        cond = np.zeros((3,3), dtype=np.float64)
        cond[0,0] = ep[i-1, 0]
        cond[1,1] = ep[i-1, 1]
        cond[2,2] = ep[i-1, 2]

        # rotation matrices Rz * Rx * Rz2
        Rz = np.array([[math.cos(alfa), -math.sin(alfa), 0.0],
                       [math.sin(alfa),  math.cos(alfa), 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, math.cos(beta), math.sin(beta)],
                       [0.0, -math.sin(beta), math.cos(beta)]], dtype=np.float64)
        R_z = np.array([[math.cos(gama), -math.sin(gama), 0.0],
                        [math.sin(gama),  math.cos(gama), 0.0],
                        [0.0, 0.0, 1.0]], dtype=np.float64)

        R = Rz @ Rx @ R_z
        sigma[i,:,:] = R @ cond @ R.T

        # permittivity from ep columns 6..14 (if zeros it's fine)
        eps_mat = np.zeros((3,3), dtype=np.float64)
        eps_mat[0,0] = ep[i-1,6]; eps_mat[0,1] = ep[i-1,7]; eps_mat[0,2] = ep[i-1,8]
        eps_mat[1,0] = ep[i-1,9]; eps_mat[1,1] = ep[i-1,10]; eps_mat[1,2] = ep[i-1,11]
        eps_mat[2,0] = ep[i-1,12]; eps_mat[2,1] = ep[i-1,13]; eps_mat[2,2] = ep[i-1,14]
        epsilon[i,:,:] = eps_mat * epsilon0

        # y_hat = sigma - i*omega*epsilon
        y_hat[i,:,:] = sigma[i,:,:] - 1j * omega * epsilon[i,:,:]

        # A matrix elimination of z
        denom = y_hat[i,2,2]
        if abs(denom) < tol:
            denom = denom + tol
        A[i,0,0] = y_hat[i,0,0] - (y_hat[i,0,2] * y_hat[i,2,0]) / denom
        A[i,0,1] = y_hat[i,0,1] - (y_hat[i,0,2] * y_hat[i,2,1]) / denom
        A[i,1,0] = y_hat[i,1,0] - (y_hat[i,1,2] * y_hat[i,2,0]) / denom
        A[i,1,1] = y_hat[i,1,1] - (y_hat[i,1,2] * y_hat[i,2,1]) / denom

        # z_hat = i*omega*mu (ep[:,15] stores mu factor)
        mu_layer = ep[i-1,15] * mu0
        z_hat[i] = 1j * omega * mu_layer

        depth[i] = ep[i-1,16]

    # thickness
    for i in range(1, n+1):
        h[i] = depth[i] - depth[i-1]

    # prepare arrays for K/Q/D/P/U...
    K1 = np.zeros(n+1, dtype=np.complex128)
    K2 = np.zeros(n+1, dtype=np.complex128)
    Q1 = np.zeros(n+1, dtype=np.complex128)
    Q2 = np.zeros(n+1, dtype=np.complex128)
    D1 = np.zeros(n+1, dtype=np.complex128)
    D2 = np.zeros(n+1, dtype=np.complex128)
    D3 = np.zeros(n+1, dtype=np.complex128)
    D4 = np.zeros(n+1, dtype=np.complex128)
    D5 = np.zeros(n+1, dtype=np.complex128)
    gama1 = np.zeros(n+1, dtype=np.complex128)
    gama2 = np.zeros(n+1, dtype=np.complex128)

    P = np.zeros((n+1,6,1), dtype=np.complex128)
    U1 = np.zeros((n+1,6,1), dtype=np.complex128)
    U2 = np.zeros((n+1,6,1), dtype=np.complex128)
    U3 = np.zeros((n+1,6,1), dtype=np.complex128)
    U4 = np.zeros((n+1,6,1), dtype=np.complex128)

    # K1,K2
    for i in range(1, n+1):
        if abs(A[i,1,0]) < tol:
            K12 = - z_hat[i] * A[i,0,0]
            K22 = - z_hat[i] * A[i,1,1]
        else:
            a = (A[i,0,0] - A[i,1,1]) * (A[i,0,0] - A[i,1,1])
            b = np.sqrt(a + 4.0 * A[i,0,1] * A[i,1,0])
            K12 = -0.5 * z_hat[i] * (A[i,0,0] + A[i,1,1] + b)
            K22 = -0.5 * z_hat[i] * (A[i,0,0] + A[i,1,1] - b)

        sqrtK12 = np.sqrt(K12)
        sqrtK22 = np.sqrt(K22)
        K1[i] = sqrtK12 if sqrtK12.real > 0 else -sqrtK12
        K2[i] = sqrtK22 if sqrtK22.real > 0 else -sqrtK22

    # Q, D, gama, P, U
    for i in range(1, n+1):
        h_n = h[i]
        z_hat_n = z_hat[i]
        A10 = A[i,1,0]
        A11 = A[i,1,1]
        k1 = K1[i]; k2 = K2[i]

        if abs(A10) < tol:
            Q1[i] = 0.0 + 0.0j
            Q2[i] = 0.0 + 0.0j
        else:
            Q1[i] = (z_hat_n * A10) / (k1*k1 + z_hat_n * A11)
            Q2[i] = (z_hat_n * A10) / (k2*k2 + z_hat_n * A11)
            if np.isnan(Q2[i]):
                Q2[i] = 0.0 + 0.0j

        e_k1 = np.exp(-2.0 * k1 * h_n)
        e_k2 = np.exp(-2.0 * k2 * h_n)
        e_k12 = np.exp(-2.0 * (k1 + k2) * h_n)

        D1[i] = 1.0 + e_k12 - e_k1 - e_k2
        D2[i] = 1.0 + e_k12 + e_k1 + e_k2
        D3[i] = 1.0 - e_k12 + e_k1 - e_k2
        D4[i] = 1.0 - e_k12 - e_k1 + e_k2
        D5[i] = 4.0 * np.exp(-(k1 + k2) * h_n)

        gama1[i] = -k1 / z_hat_n
        gama2[i] = -k2 / z_hat_n

        # q related
        if abs(A10) < tol:
            q1 = 0.0 + 0.0j
            q2 = 0.0 + 0.0j
            q  = 0.0 + 0.0j
        else:
            q1 = Q1[i]
            q2 = 1.0 / Q2[i] if Q2[i] != 0 else 0.0 + 0.0j
            q  = q1 * q2

        r1 = gama1[i]; r2 = gama2[i]
        d1 = D1[i]; d2 = D2[i]; d3 = D3[i]; d4 = D4[i]; d5 = D5[i]

        P[i,0,0] = r1 * r2 * d1 * (q - 1.0)
        P[i,1,0] = q1 * (r2*d3 - r1*d4)
        P[i,2,0] = q * r2 * d3 - r1 * d4
        P[i,3,0] = r2 * d3 - q * r1 * d4
        P[i,4,0] = q2 * (r2*d3 - r1*d4)
        P[i,5,0] = (q - 1.0) * d2

        qq = (q - 1.0) * r1 * r2
        if qq == 0:
            qq = 1.0 + 0.0j

        # U1
        U1[i,0,0] = (r2*d3 - r1*d4) * q2
        U1[i,1,0] = (q*d1*(r1*r1 + r2*r2) + ((q*q + 1.0)*d5 - 2*q*d2)*r1*r2) / qq
        U1[i,2,0] = (d1*(q*r2*r2 + r1*r1) + (q+1.0)*(d5 - d2)*r1*r2) * q2 / qq
        U1[i,3,0] = (d1*(q*r1*r1 + r2*r2) + (q+1.0)*(d5 - d2)*r1*r2) * q2 / qq
        U1[i,4,0] = (d1*(r1*r1 + r2*r2) + 2*(d5 - d2)*r1*r2) * q2*q2 / qq
        U1[i,5,0] = (r2*d4 - r1*d3) / (r1*r2) * q2

        # U2
        U2[i,0,0] = q*r1*d4 - r2*d3
        U2[i,1,0] = (-q1*((q*r1*r1 + r2*r2)*d1 + (d5 - d2)*(q+1.0)*r1*r2)) / qq
        U2[i,2,0] = ((q*q + 1.0)*r1*r2*d2 - q*((r1*r1 + r2*r2)*d1 + 2*r1*r2*d5)) / qq
        U2[i,3,0] = (-d1*(q*q*r1*r1 + r2*r2) - 2*q*r1*r2*(d5 - d2)) / qq
        U2[i,4,0] = (-d1*(q*r1*r1 + r2*r2) - (d5 - d2)*(q+1.0)*r1*r2) * q2 / qq
        U2[i,5,0] = (q*r1*d3 - r2*d4) / (r1*r2)

        # U3
        U3[i,0,0] = r1*d4 - q*r2*d3
        U3[i,1,0] = -(q1*((r1*r1 + q*r2*r2)*d1 + (d5 - d2)*(q+1.0)*r1*r2)) / qq
        U3[i,2,0] = -(d1*(r1*r1 + q*q*r2*r2) + 2*q*r1*r2*(d5 - d2)) / qq
        U3[i,3,0] = ((q*q + 1.0)*r1*r2*d2 - q*((r1*r1 + r2*r2)*d1 + 2*r1*r2*d5)) / qq
        U3[i,4,0] = -(d1*(r1*r1 + q*r2*r2) + (d5 - d2)*(q+1.0)*r1*r2) * q2 / qq
        U3[i,5,0] = (r1*d3 - q*r2*d4) / (r1*r2)

        # U4
        U4[i,0,0] = q1*(r2*d3 - r1*d4)
        U4[i,1,0] = (q1*q1*((r1*r1 + r2*r2)*d1 + 2*r1*r2*(d5 - d2))) / qq
        U4[i,2,0] = (q1*((r1*r1 + q*r2*r2)*d1 + (d5 - d2)*(q+1.0)*r1*r2)) / qq
        U4[i,3,0] = (q1*((q*r1*r1 + r2*r2)*d1 + (d5 - d2)*(q+1.0)*r1*r2)) / qq
        U4[i,4,0] = (q*d1*(r1*r1 + r2*r2) + r1*r2*((q*q + 1.0)*d5 - 2*q*d2)) / qq
        U4[i,5,0] = q1*(r2*d4 - r1*d3) / (r1*r2)

    # 3) bottom layer impedance and upward recursion
    Z = np.zeros((n,2,2), dtype=np.complex128)

    # ensure q,q1,q2 exist
    try:
        q
    except NameError:
        q = 0.0 + 0.0j
    try:
        q1
    except NameError:
        q1 = 0.0 + 0.0j
    try:
        q2
    except NameError:
        q2 = 0.0 + 0.0j

    denom_base = (gama1[n] * gama2[n] * (q - 1.0))
    if denom_base == 0:
        denom_base = 1.0 + 0.0j

    ZN = np.zeros((2,2), dtype=np.complex128)
    ZN[0,0] = (gama2[n] - gama1[n]) * q2
    ZN[0,1] = q * gama1[n] - gama2[n]
    ZN[1,0] = gama1[n] - q * gama2[n]
    ZN[1,1] = q1 * (gama2[n] - gama1[n])

    Z[n-1,:,:] = ZN / denom_base

    for i in range(n-1, 0, -1):
        V = np.zeros((6,1), dtype=np.complex128)
        V[0,0] = Z[i,0,0] * Z[i,1,1] - Z[i,0,1] * Z[i,1,0]
        V[1,0] = Z[i,0,0]
        V[2,0] = Z[i,0,1]
        V[3,0] = Z[i,1,0]
        V[4,0] = Z[i,1,1]
        V[5,0] = 1.0 + 0.0j

        det = (P[i].T @ V).item()
        M = np.zeros((2,2), dtype=np.complex128)
        M[0,0] = (U1[i].T @ V).item()
        M[0,1] = (U2[i].T @ V).item()
        M[1,0] = (U3[i].T @ V).item()
        M[1,1] = (U4[i].T @ V).item()

        if det == 0:
            det = 1.0 + 0.0j

        Z[i-1,:,:] = M / det

    Zsurf = Z[0,:,:]

    # 4) apparent resistivity & phase
    App = np.zeros((2,2), dtype=np.float64)
    Phase = np.zeros((2,2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            val = Zsurf[i,j]
            App[i,j] = (abs(val) * abs(val)) / (2.0 * math.pi * f * mu0)
            ph = math.atan2(val.imag, val.real) * 180.0 / math.pi
            if 90 <= ph <= 180:
                ph = ph - 90.0
            elif -180 <= ph <= -90:
                ph = ph + 180.0
            elif -90 <= ph <= 0:
                ph = ph + 90.0
            Phase[i,j] = ph

    return App, Phase, Zsurf


def Frequency_Domain_MT_Modeling(data_type, frequencies, depth_arr, true_model, n_jobs=None, threshold=200):
    """
    智能调度入口，返回格式尽量与原函数保持一致
    data_type: 1..5 参考你原始函数
    frequencies: 1D array-like
    depth_arr: 与原来 deepth 保持同结构 (例如 shape (1, n+1))
    true_model: shape (3, n_layers): [rho_x', rho_y', alfa(deg) maybe]
    n_jobs: None->自动选择 (cpu_count()-1), or int
    threshold: 频率点阈值，≤threshold 用单进程，否则并行
    """
    # build ep array identical于原始
    n_l = true_model.shape[1]
    ep = np.zeros((n_l,17), dtype=np.float64)
    for i in range(n_l):
        if i == (n_l - 1):
            ep[i,16] = float('inf')
        else:
            ep[i,16] = depth_arr[0, i+1] if depth_arr is not None else float('inf')

        ep[i,0] = 1.0 / true_model[0, i]
        ep[i,1] = 1.0 / true_model[1, i]
        ep[i,2] = 1.0 / true_model[0, i]
        # ep[i,3] = true_model[2, i]   # alfa in degrees
        ep[i,3] = 0   # alfa in degrees

        ep[i,15] = 1.0

    freqs = np.asarray(frequencies, dtype=np.float64)
    n_f = freqs.size

    # auto n_jobs
    if n_jobs is None:
        cpu = multiprocessing.cpu_count()
        n_jobs = max(1, cpu - 1)

    results = None
    start = time.time()
    if n_f <= threshold:
        # print(f"[INFO] Using SERIAL (n_freq={n_f} <= threshold={threshold})")
        # serial compute, directly call MT1D
        results = [MT1D(n_l, ep, float(f)) for f in freqs]
    else:
        # print(f"[INFO] Using PARALLEL (n_freq={n_f} > threshold={threshold}), n_jobs={n_jobs}")
        # set globals so each worker doesn't pickle ep per task
        global _EP_GLOBAL, _N_GLOBAL
        _EP_GLOBAL = ep
        _N_GLOBAL = n_l
        # execute parallel: pass only frequency values to workers
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_mt_worker)(float(f)) for f in freqs
        )
    end = time.time()
    # print(f"[INFO] Computation done in {end-start:.3f} s")

    # pack outputs to match your original function formats
    Zxy = np.zeros((2, n_f), dtype=np.float64)
    Zyx = np.zeros((2, n_f), dtype=np.float64)
    Zxy_yx = np.zeros((4, n_f), dtype=np.float64)
    app_pha_xy = np.zeros((2, n_f), dtype=np.float64)
    app_pha_yx = np.zeros((2, n_f), dtype=np.float64)
    app_pha = np.zeros((4, n_f), dtype=np.float64)
    app = np.zeros((2, n_f), dtype=np.float64)

    for k in range(n_f):
        App, Ph, Zsurf = results[k]
        Zxy[0,k] = Zsurf[0,1].real
        Zxy[1,k] = Zsurf[0,1].imag
        Zyx[0,k] = Zsurf[1,0].real
        Zyx[1,k] = Zsurf[1,0].imag

        Zxy_yx[0,k] = Zxy[0,k]
        Zxy_yx[1,k] = Zxy[1,k]
        Zxy_yx[2,k] = Zyx[0,k]
        Zxy_yx[3,k] = Zyx[1,k]

        app_pha_xy[0,k] = App[0,1]
        app_pha_xy[1,k] = Ph[0,1]
        app_pha_yx[0,k] = App[1,0]
        app_pha_yx[1,k] = Ph[1,0]

        app_pha[0,k] = App[0,1]
        app_pha[1,k] = Ph[0,1]
        app_pha[2,k] = App[1,0]
        app_pha[3,k] = Ph[1,0]

        app[0,k] = App[0,1]
        app[1,k] = App[1,0]

    if data_type == 1.0:
        return Zxy
    elif data_type == 2.0:
        return app_pha_xy
    elif data_type == 3.0:
        return Zxy_yx
    elif data_type == 4.0:
        return app_pha
    elif data_type == 5.0:
        return app
    else:
        return {
            "Zxy": Zxy, "Zyx": Zyx, "Zxy_yx": Zxy_yx,
            "app_pha_xy": app_pha_xy, "app_pha_yx": app_pha_yx,
            "app_pha": app_pha, "app": app
        }



# ========== 测试 / 使用示例 ==========
if __name__ == "__main__":

    t0 = time.time()


    for i in range(1):
        # 简单示例模型（与原示例保持兼容）
        freqs = np.logspace(-3, 3, 32)   # 50 frequency points
        # true_model shape (3, n_layers)
        # we keep the same mapping as original: true_model[0,i] and [1,i] are resistivities
        true_model = np.array([
            [100.0, 10.0, 500.0],   # rho_x' per layer
            [50.0, 30.0, 500.0],    # rho_y' per layer
            [30.0, 50.0, 0.0]         # alfaz (degrees) - you can set per-layer Euler angles here
        ])
        # depth array similar to your deepth input: 2D maybe; we only need indexing [0,i+1]
        deepth = np.zeros((1, true_model.shape[1] + 1))
        deepth[0,1] = 500.0
        deepth[0,2] = 1500.0
        deepth[0,3] = float('inf')

        out = Frequency_Domain_MT_Modeling(4.0, freqs, deepth, true_model, n_jobs=None, threshold=500)

        print(out.shape)    # (4, n_freq), first two rows: Zxy app & phase, last two: Zyx app & phase
        # print(out.T)


    t1 = time.time()
    print("Computed for %d frequencies in %.3f s" % (len(freqs), t1 - t0))