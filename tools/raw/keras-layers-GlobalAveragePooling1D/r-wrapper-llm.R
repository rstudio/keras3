#' Global average pooling operation for temporal data.
#'
#' @description
#'
#' # Call Arguments
#' - `inputs`: A 3D tensor.
#' - `mask`: Binary tensor of shape `(batch_size, steps)` indicating whether
#'     a given step should be masked (excluded from the average).
#'
#' # Input Shape
#' - If `data_format='channels_last'`:
#'     3D tensor with shape:
#'     `(batch_size, steps, features)`
#' - If `data_format='channels_first'`:
#'     3D tensor with shape:
#'     `(batch_size, features, steps)`
#'
#' # Output Shape
#' - If `keepdims=False`:
#'     2D tensor with shape `(batch_size, features)`.
#' - If `keepdims=True`:
#'     - If `data_format="channels_last"`:
#'         3D tensor with shape `(batch_size, 1, features)`
#'     - If `data_format="channels_first"`:
#'         3D tensor with shape `(batch_size, features, 1)`
#'
#' # Examples
#' ```python
#' x = np.random.rand(2, 3, 4)
#' y = keras.layers.GlobalAveragePooling1D()(x)
#' y.shape
#' # (2, 4)
#' ```
#'
#' @param data_format string, either `"channels_last"` or `"channels_first"`.
#'     The ordering of the dimensions in the inputs. `"channels_last"`
#'     corresponds to inputs with shape `(batch, steps, features)`
#'     while `"channels_first"` corresponds to inputs with shape
#'     `(batch, features, steps)`. It defaults to the `image_data_format`
#'     value found in your Keras config file at `~/.keras/keras.json`.
#'     If you never set it, then it will be `"channels_last"`.
#' @param keepdims A boolean, whether to keep the temporal dimension or not.
#'     If `keepdims` is `False` (default), the rank of the tensor is
#'     reduced for spatial dimensions. If `keepdims` is `True`, the
#'     temporal dimension are retained with length 1.
#'     The behavior is the same as for `tf.reduce_mean` or `np.mean`.
#' @param object Object to compose the layer with. A tensor, array, or sequential model.
#' @param ... Passed on to the Python callable
#'
#' @export
#' @family pooling layers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling1D>
layer_global_average_pooling_1d <-
function (object, data_format = NULL, keepdims = FALSE, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$GlobalAveragePooling1D, object,
        args)
}
