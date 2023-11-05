Apply additive zero-centered Gaussian noise.

This is useful to mitigate overfitting
(you could see it as a form of random data augmentation).
Gaussian Noise (GS) is a natural choice as corruption process
for real valued inputs.

As it is a regularization layer, it is only active at training time.

Args:
    stddev: Float, standard deviation of the noise distribution.
    seed: Integer, optional random seed to enable deterministic behavior.

Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
        training mode (adding noise) or in inference mode (doing nothing).
