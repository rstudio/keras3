#' Performs spectral normalization on the weights of a target layer.
#'
#' @description
#' This wrapper controls the Lipschitz constant of the weights of a layer by
#' constraining their spectral norm, which can stabilize the training of GANs.
#'
#' # Examples
#' Wrap `keras.layers.Conv2D`:
#' ```python
#' x = np.random.rand(1, 10, 10, 1)
#' conv2d = SpectralNormalization(keras.layers.Conv2D(2, 2))
#' y = conv2d(x)
#' y.shape
#' # (1, 9, 9, 2)
#' ```
#'
#' Wrap `keras.layers.Dense`:
#' ```python
#' x = np.random.rand(1, 10, 10, 1)
#' dense = SpectralNormalization(keras.layers.Dense(10))
#' y = dense(x)
#' y.shape
#' # (1, 10, 10, 10)
#' ```
#'
#' # Reference
#' - [Spectral Normalization for GAN](https://arxiv.org/abs/1802.05957).
#'
#' @param layer
#' A `keras.layers.Layer` instance that
#' has either a `kernel` (e.g. `Conv2D`, `Dense`...)
#' or an `embeddings` attribute (`Embedding` layer).
#'
#' @param power_iterations
#' int, the number of iterations during normalization.
#'
#' @param ...
#' Base wrapper keyword arguments.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @export
#' @family normalization layers
#' @family layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/SpectralNormalization>
layer_spectral_normalization <-
function (object, layer, power_iterations = 1L, ...)
{
}
