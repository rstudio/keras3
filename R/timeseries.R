
#' Utility function for generating batches of temporal data.
#'
#' @param data Object containing consecutive data points (timesteps). The data
#'   should be 2D, and axis 1 is expected to be the time dimension.
#' @param targets Targets corresponding to timesteps in `data`.
#'   It should have same length as `data`.
#' @param length Length of the output sequences (in number of timesteps).
#' @param sampling_rate Period between successive individual timesteps
#'   within sequences. For rate `r`, timesteps `data[i]`, `data[i-r]`, ... `data[i - length]`
#'   are used for create a sample sequence.
#' @param stride Period between successive output sequences.
#'  For stride `s`, consecutive output samples would
#'  be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
#' @param start_index,end_index Data points earlier than `start_index`
#'   or later than `end_index` will not be used in the output sequences.
#'   This is useful to reserve part of the data for test or validation.
#' @param shuffle  Whether to shuffle output samples,
#'   or instead draw them in chronological order.
#' @param reverse Boolean: if `true`, timesteps in each output sample will be
#'   in reverse chronological order.
#' @param batch_size Number of timeseries samples in each batch
#'   (except maybe the last one).
#'
#' @return An object that can be passed to generator based training
#'   functions (e.g. [fit_generator()]).ma
#'
#' @export
timeseries_generator <- function(data, targets, length, sampling_rate = 1,
                                 stride = 1, start_index = 0, end_index = NULL,
                                 shuffle = FALSE, reverse = FALSE, batch_size = 128) {
  keras$preprocessing$sequence$TimeseriesGenerator(
    data = keras_array(data),
    targets = keras_array(targets),
    length = as.integer(length),
    sampling_rate = as.integer(sampling_rate),
    stride = as.integer(stride),
    start_index = as.integer(start_index),
    end_index = as_nullable_integer(end_index),
    shuffle = shuffle,
    reverse = reverse,
    batch_size = as.integer(batch_size)
  )
}
