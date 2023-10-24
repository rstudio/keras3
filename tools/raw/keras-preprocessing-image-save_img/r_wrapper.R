#' Saves an image stored as a NumPy array to a path or file object.
#'
#' @description
#'
#' @param path Path or file object.
#' @param x NumPy array.
#' @param data_format Image data format, either `"channels_first"` or
#'     `"channels_last"`.
#' @param file_format Optional file format override. If omitted, the format to
#'     use is determined from the filename extension. If a file object was
#'     used instead of a filename, this parameter should always be used.
#' @param scale Whether to rescale image values to be within `[0, 255]`.
#' @param ... Additional keyword arguments passed to `PIL.Image.save()`.
#'
#' @export
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/save_img>
image_array_save <-
function (x, path, data_format = NULL, file_format = NULL, scale = TRUE,
    ...)
{
    args <- capture_args2(NULL)
    do.call(keras$preprocessing$image$save_img, args)
}
