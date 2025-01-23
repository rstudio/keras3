
#' Saves an image stored as an array to a path or file object.
#'
#' @param path
#' Path or file object.
#'
#' @param x
#' An array.
#'
#' @param data_format
#' Image data format, either `"channels_first"` or
#' `"channels_last"`.
#'
#' @param file_format
#' Optional file format override. If omitted, the format to
#' use is determined from the filename extension. If a file object was
#' used instead of a filename, this parameter should always be used.
#'
#' @param scale
#' Whether to rescale image values to be within `[0, 255]`.
#'
#' @param ...
#' Additional keyword arguments passed to `PIL.Image.save()`.
#'
#' @returns Called primarily for side effects. The input `x` is returned,
#'   invisibly, to enable usage with the pipe.
#' @export
#' @family image utils
#' @family utils
#' @seealso
#' + <https://keras.io/api/data_loading/image#saveimg-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/save_img>
#' @tether keras.utils.save_img
image_array_save <-
function (x, path, data_format = NULL, file_format = NULL, scale = TRUE,
          ...)
{
  args <- capture_args()
  do.call(keras$utils$save_img, args)
  invisible(x)
}


#' Converts a 3D array to a PIL Image instance.
#'
#' @description
#'
#' # Example
#'
#' ```{r}
#' img <- array(runif(30000), dim = c(100, 100, 3))
#' pil_img <- image_from_array(img)
#' pil_img
#' ```
#'
#' @returns
#' A PIL Image instance.
#'
#' @param x
#' Input data, in any form that can be converted to an array.
#'
#' @param data_format
#' Image data format, can be either `"channels_first"` or
#' `"channels_last"`. Defaults to `NULL`, in which case the global
#' setting `config_image_data_format()` is used (unless you
#' changed it, it defaults to `"channels_last"`).
#'
#' @param scale
#' Whether to rescale the image such that minimum and maximum values
#' are 0 and 255 respectively. Defaults to `TRUE`.
#'
#' @param dtype
#' Dtype to use. `NULL` means the global setting
#' `config_floatx()` is used (unless you changed it, it
#' defaults to `"float32"`). Defaults to `NULL`.
#'
#' @export
#' @family image utils
#' @family utils
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/array_to_img>
#' @tether keras.utils.array_to_img
image_from_array <-
function (x, data_format = NULL, scale = TRUE, dtype = NULL)
{
  args <- capture_args()
  do.call(keras$utils$array_to_img, args)
}


#' Loads an image into PIL format.
#'
#' @description
#'
#' # Example
#'
#' ```{r}
#' image_path <- get_file(origin = "https://www.r-project.org/logo/Rlogo.png")
#' (image <- image_load(image_path))
#'
#' input_arr <- image_to_array(image)
#' str(input_arr)
#' input_arr %<>% array_reshape(dim = c(1, dim(input_arr))) # Convert single image to a batch.
#' ```
#'
#'
#' ```{r, eval = FALSE}
#' model |> predict(input_arr)
#' ```
#'
#' @returns
#' A PIL Image instance.
#'
#' @param path
#' Path to image file.
#'
#' @param color_mode
#' One of `"grayscale"`, `"rgb"`, `"rgba"`. Default: `"rgb"`.
#' The desired image format.
#'
#' @param target_size
#' Either `NULL` (default to original size) or tuple of ints
#' `(img_height, img_width)`.
#'
#' @param interpolation
#' Interpolation method used to resample the image if the
#' target size is different from that of the loaded image. Supported
#' methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
#' If PIL version 1.1.3 or newer is installed, `"lanczos"`
#' is also supported. If PIL version 3.4.0 or newer is installed,
#' `"box"` and `"hamming"` are also
#' supported. By default, `"nearest"` is used.
#'
#' @param keep_aspect_ratio
#' Boolean, whether to resize images to a target
#' size without aspect ratio distortion. The image is cropped in
#' the center with target aspect ratio before resizing.
#'
#' @export
#' @family image utils
#' @family utils
#' @seealso
#' + <https://keras.io/api/data_loading/image#loadimg-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/load_img>
#' @tether keras.utils.load_img
image_load <-
function (path, color_mode = "rgb", target_size = NULL, interpolation = "nearest",
          keep_aspect_ratio = FALSE)
{
  args <- capture_args(list(target_size = as_integer_tuple))
  do.call(keras$utils$load_img, args)
}


#' Converts a PIL Image instance to a matrix.
#'
#' @description
#'
#' # Example
#'
#' ```{r}
#' image_path <- get_file(origin = "https://www.r-project.org/logo/Rlogo.png")
#' (img <- image_load(image_path))
#'
#' array <- image_to_array(img)
#' str(array)
#' ```
#'
#' @returns
#' A 3D array.
#'
#' @param img
#' Input PIL Image instance.
#'
#' @param data_format
#' Image data format, can be either `"channels_first"` or
#' `"channels_last"`. Defaults to `NULL`, in which case the global
#' setting `config_image_data_format()` is used (unless you
#' changed it, it defaults to `"channels_last"`).
#'
#' @param dtype
#' Dtype to use. `NULL` means the global setting
#' `config_floatx()` is used (unless you changed it, it
#' defaults to `"float32"`).
#'
#' @export
#' @family image utils
#' @family utils
#' @seealso
#' + <https://keras.io/api/data_loading/image#imgtoarray-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/img_to_array>
#' @tether keras.utils.img_to_array
image_to_array <-
function (img, data_format = NULL, dtype = NULL)
{
  args <- capture_args()
  do.call(keras$utils$img_to_array, args)
}
