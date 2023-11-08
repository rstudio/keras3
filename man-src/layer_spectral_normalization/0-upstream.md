keras.layers.SpectralNormalization
__signature__
(layer, power_iterations=1, **kwargs)
__doc__
Performs spectral normalization on the weights of a target layer.

This wrapper controls the Lipschitz constant of the weights of a layer by
constraining their spectral norm, which can stabilize the training of GANs.

Args:
    layer: A `keras.layers.Layer` instance that
        has either a `kernel` (e.g. `Conv2D`, `Dense`...)
        or an `embeddings` attribute (`Embedding` layer).
    power_iterations: int, the number of iterations during normalization.
    **kwargs: Base wrapper keyword arguments.

Examples:

Wrap `keras.layers.Conv2D`:
>>> x = np.random.rand(1, 10, 10, 1)
>>> conv2d = SpectralNormalization(keras.layers.Conv2D(2, 2))
>>> y = conv2d(x)
>>> y.shape
(1, 9, 9, 2)

Wrap `keras.layers.Dense`:
>>> x = np.random.rand(1, 10, 10, 1)
>>> dense = SpectralNormalization(keras.layers.Dense(10))
>>> y = dense(x)
>>> y.shape
(1, 10, 10, 10)

Reference:

- [Spectral Normalization for GAN](https://arxiv.org/abs/1802.05957).
