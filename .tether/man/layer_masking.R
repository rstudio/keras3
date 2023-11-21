#' Masks a sequence by using a mask value to skip timesteps.
#'
#' @description
#' For each timestep in the input tensor (dimension #1 in the tensor),
#' if all values in the input tensor at that timestep
#' are equal to `mask_value`, then the timestep will be masked (skipped)
#' in all downstream layers (as long as they support masking).
#'
#' If any downstream layer does not support masking yet receives such
#' an input mask, an exception will be raised.
#'
#' # Examples
#' Consider a NumPy data array `x` of shape `(samples, timesteps, features)`,
#' to be fed to an LSTM layer. You want to mask timestep #3 and #5 because you
#' lack data for these timesteps. You can:
#'
#' - Set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
#' - Insert a `Masking` layer with `mask_value=0.` before the LSTM layer:
#'
#' ```python
#' samples, timesteps, features = 32, 10, 8
#' inputs = np.random.random([samples, timesteps, features]).astype(np.float32)
#' inputs[:, 3, :] = 0.
#' inputs[:, 5, :] = 0.
#'
#' model = keras.models.Sequential()
#' model.add(keras.layers.Masking(mask_value=0.)
#' model.add(keras.layers.LSTM(32))
#' output = model(inputs)
#' # The time step 3 and 5 will be skipped from LSTM calculation.
#' ```
#'
#' # Note
#' in the Keras masking convention, a masked timestep is denoted by
#' a mask value of `False`, while a non-masked (i.e. usable) timestep
#' is denoted by a mask value of `True`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' Passed on to the Python callable
#'
#' @param mask_value
#' see description
#'
#' @export
#' @family core layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/core_layers/masking#masking-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Masking>
layer_masking <-
function (object, mask_value = 0, ...)
{
}
