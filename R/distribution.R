#' Count available devices of one type
#'
#' When `device_type` is not provided, Keras counts devices of the default
#' available type. Device types are not mixed in a single count.
#' Keras 3.15.1 provides the backend implementation for JAX.
#'
#' # Examples
#'
#' ```{r, eval = FALSE}
#' distributed_get_device_count("cpu")
#' ```
#'
#' @param device_type String, one of `"cpu"`, `"gpu"`, or `"tpu"`. If `NULL`,
#'   Keras counts GPU or TPU devices when available and otherwise counts CPU
#'   devices.
#'
#' @returns An integer number of devices.
#' @export
#' @family distribution
#' @tether keras.distribution.get_device_count
distributed_get_device_count <-
function(device_type = NULL)
{
  args <- capture_args()
  do.call(keras$distribution$get_device_count, args)
}
