# PCMT-Net-for-1-D-Anisotropic-Magnetotelluric-Inversion
A better method for fabricating one-dimensional anisotropic samples and an inversion model

## Generation and pairing of GRF samples.ipynb
A better approach is to use this program to generate physically consistent samples, and then pair the samples using the Pearson coefficient to construct a one-dimensional anisotropic sample group (64 one-dimensional samples form one group).

## Structure-Preserving Layered Parameterization.ipynb
Using this program, convert the curve type samples into layered samples with similar structures, which are used to enrich the sample construction and create a high-quality dataset.

## Parallel Frequency-wise Forward Computation.ipynb
Use this program to perform forward calculations (in parallel) on the prepared one-dimensional anisotropic electrical models to generate samples for training.

## PCMT_Net_train.ipynb
After dividing the sample set into certain proportions, substitute it into the program for training. After training is completed, use the test set to evaluate the model's accuracy.

## test experiment.ipynb
Load the trained model weights and use them for experimental testing and error statistics for the anisotropic sample production method and inversion model.

In addition, the repository provides the core forward-modeling program MT1D_Forward.py. To ensure proper execution, this program should be placed in the same directory as Parallel Frequency-wise Forward Computation.ipynb.

A sample dataset is also included in the repository to illustrate the data format and the input–output structure used in this study. Specifically, X_all_10000.npy contains the MT response data, including the apparent resistivity and phase in both the xy and yx polarization modes, with a data shape of (10000, 4, 64). The file Y_all_10000.npy contains the corresponding anisotropic resistivity models, including the resistivity distributions in the X and Y directions, with a data shape of (10000, 2, 64). new003.pth represents the weights obtained after training the PCMT-Net model.
