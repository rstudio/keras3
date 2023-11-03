Apply multiplicative 1-centered Gaussian noise.

As it is a regularization layer, it is only active at training time.

Args:
    rate: Float, drop probability (as with `Dropout`).
        The multiplicative noise will have
        standard deviation `sqrt(rate / (1 - rate))`.
    seed: Integer, optional random seed to enable deterministic behavior.

Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
