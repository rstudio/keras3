#' Converts a 3D NumPy array to a PIL Image instance.
#'
#' @description
#'
#' # Usage
#' ```python
#' from PIL import Image
#' img = np.random.random(size=(100, 100, 3))
#' pil_img = keras.utils.array_to_img(img)
#' ```
#'
#' # Returns
#'     A PIL Image instance.
#'
#' @param x Input data, in any form that can be converted to a NumPy array.
#' @param data_format Image data format, can be either `"channels_first"` or
#'     `"channels_last"`. Defaults to `None`, in which case the global
#'     setting `keras.backend.image_data_format()` is used (unless you
#'     changed it, it defaults to `"channels_last"`).
#' @param scale Whether to rescale the image such that minimum and maximum values
#'     are 0 and 255 respectively. Defaults to `True`.
#' @param dtype Dtype to use. `None` means the global setting
#'     `keras.backend.floatx()` is used (unless you changed it, it
#'     defaults to `"float32"`). Defaults to `None`.
#'
#' @export
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/array_to_img>
image_from_array <-
function (x, data_format = NULL, scale = TRUE, dtype = NULL)
{
    args <- capture_args2(NULL)
    do.call(keras$preprocessing$image$array_to_img, args)
}
