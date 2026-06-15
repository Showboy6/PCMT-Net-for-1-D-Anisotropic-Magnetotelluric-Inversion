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
