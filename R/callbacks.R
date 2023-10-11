

#' (Deprecated) Base R6 class for Keras callbacks
#'
#' New custom callbacks implemented as R6 classes are encouraged to inherit from
#' `keras$callbacks$Callback` directly.
#'
#' @docType class
#'
#' @format An [R6Class] generator object
#'
#' @field params Named list with training parameters (eg. verbosity, batch size, number of epochs...).
#' @field model Reference to the Keras model being trained.
#'
#' @section Methods:
#' \describe{
#'  \item{\code{on_epoch_begin(epoch, logs)}}{Called at the beginning of each epoch.}
#'  \item{\code{on_epoch_end(epoch, logs)}}{Called at the end of each epoch.}
#'  \item{\code{on_batch_begin(batch, logs)}}{Called at the beginning of each batch.}
#'  \item{\code{on_batch_end(batch, logs)}}{Called at the end of each batch.}
#'  \item{\code{on_train_begin(logs)}}{Called at the beginning of training.}
#'  \item{\code{on_train_end(logs)}}{Called at the end of training.}
#' }
#'
#' @details  The `logs` named list that callback methods take as argument will
#' contain keys for quantities relevant to the current batch or epoch.
#'
#' Currently, the `fit.keras.models.model.Model()` method for sequential
#' models will include the following quantities in the `logs` that
#' it passes to its callbacks:
#'
#' - `on_epoch_end`: logs include `acc` and `loss`, and optionally include `val_loss` (if validation is enabled in `fit`), and `val_acc` (if validation and accuracy monitoring are enabled).
#' - `on_batch_begin`: logs include `size`, the number of samples in the current batch.
#' - `on_batch_end`: logs include `loss`, and optionally `acc` (if accuracy monitoring is enabled).
#'
#' @return [KerasCallback].
#' @keywords internal
#' @examples
#' \dontrun{
#' library(keras)
#'
#' LossHistory <- R6::R6Class("LossHistory",
#'   inherit = KerasCallback,
#'
#'   public = list(
#'
#'     losses = NULL,
#'
#'     on_batch_end = function(batch, logs = list()) {
#'       self$losses <- c(self$losses, logs[["loss"]])
#'     }
#'   )
#' )
#' }
#' @export
KerasCallback <- R6Class("KerasCallback",

  public = list(

    params = NULL,
    model = NULL,

    set_context = function(params = NULL, model = NULL) {
      self$params <- params
      self$model <- model
    },

    on_epoch_begin = function(epoch, logs = NULL) {

    },

    on_epoch_end = function(epoch, logs = NULL) {

    },

    on_batch_begin = function(batch, logs = NULL) {

    },

    on_batch_end = function(batch, logs = NULL) {

    },

    on_train_begin = function(logs = NULL) {

    },

    on_train_end = function(logs = NULL) {

    },

    on_predict_batch_begin = function(batch, logs = NULL) {

    },

    on_predict_batch_end = function(batch, logs = NULL) {

    },

    on_predict_begin = function(logs = NULL) {

    },

    on_predict_end = function(logs = NULL) {

    },

    on_test_batch_begin = function(batch, logs = NULL) {

    },

    on_test_batch_end = function(batch, logs = NULL) {

    },

    on_test_begin = function(logs = NULL) {

    },

    on_test_end = function(logs = NULL) {

    },

    on_train_batch_begin = function(batch, logs = NULL) {

    },

    on_train_batch_end = function(batch, logs = NULL) {

    }

  )
)

normalize_callbacks_with_metrics <- function(view_metrics, initial_epoch, callbacks) {

  # if callbacks isn't a list then make it one
  if (!is.null(callbacks) && !is.list(callbacks))
    callbacks <- list(callbacks)

  # always include the metrics callback
  if (tensorflow::tf_version() >= "2.2.0")
    metrics_callback <- KerasMetricsCallbackV2$new(view_metrics, initial_epoch)
  else
    metrics_callback <- KerasMetricsCallback$new(view_metrics)

  callbacks <- append(callbacks, metrics_callback)

  normalize_callbacks(callbacks)
}

warn_callback <- function(callback) {

  new_callbacks <- c("on_predict_batch_begin", "on_predict_batch_end",
    "on_predict_begin", "on_predict_end",
    "on_test_batch_begin", "on_test_batch_end",
    "on_test_begin", "on_test_end",
    "on_train_batch_begin", "on_train_batch_end"
    )

  lapply(new_callbacks, function(x) {


    if (!(get_keras_implementation() == "tensorflow" &&
          tensorflow::tf_version() >= "2.0")) {

      if (inherits(callback, "KerasCallback")) {

        # workaround to find out if the body is empty as expected.
        bdy <- paste(as.character(body(callback[[x]])), collapse = "")

        if (is.null(body) || bdy != "{") {
          warning("Callback '", x, "' only works with Keras TensorFlow",
                  " implementation and Tensorflow >= 2.0")
        }

      } else if (inherits(callback, "list")) {

        if (!is.null(callback[[x]])) {
          warning("Callback '", x, "' only works with Keras TensorFlow",
                  " implementation and Tensorflow >= 2.0")
        }

      }

    }

  })

  invisible(NULL)
}

normalize_callbacks <- function(callbacks) {

  # if callbacks isn't a list then make it one
  if (!is.null(callbacks) && !is.list(callbacks))
    callbacks <- list(callbacks)

  # import callback utility module
  python_path <- system.file("python", package = "keras")
  tools <- import_from_path("kerastools", path = python_path)

  # convert R callbacks to Python and check whether the user
  # has already included the tensorboard callback
  have_tensorboard_callback <- FALSE
  callbacks <- lapply(callbacks, function(callback) {

    warn_callback(callback)

    # track whether we have a TensorBoard callback
    if (inherits(callback, "keras.callbacks.TensorBoard"))
      have_tensorboard_callback <<- TRUE

    if (inherits(callback, "KerasCallback")) {

      args <- list(
        r_set_context = callback$set_context,
        r_on_epoch_begin = callback$on_epoch_begin,
        r_on_epoch_end = callback$on_epoch_end,
        r_on_train_begin = callback$on_train_begin,
        r_on_train_end = callback$on_train_end,
        r_on_batch_begin = callback$on_batch_begin,
        r_on_batch_end = callback$on_batch_end,
        r_on_predict_batch_begin = callback$on_predict_batch_begin,
        r_on_predict_batch_end = callback$on_predict_batch_end,
        r_on_predict_begin = callback$on_predict_begin,
        r_on_predict_end = callback$on_predict_end,
        r_on_test_batch_begin = callback$on_test_batch_begin,
        r_on_test_batch_end = callback$on_test_batch_end,
        r_on_test_begin = callback$on_test_begin,
        r_on_test_end = callback$on_test_end,
        r_on_train_batch_begin = callback$on_train_batch_begin,
        r_on_train_batch_end = callback$on_train_batch_end
      )

      # on_batch_* -> on_train_batch_*
      if (!isTRUE(all.equal(callback$on_batch_begin, empty_fun))) {
        args$r_on_train_batch_begin <- callback$on_batch_begin
      }

      if (!isTRUE(all.equal(callback$on_batch_end, empty_fun))) {
        args$r_on_train_batch_end <- callback$on_batch_end
      }

      # create a python callback to map to our R callback
      do.call(tools$callback$RCallback, args)
    } else {
      callback
    }
  })

  # add the tensorboard callback if necessary
  if ((nzchar(Sys.getenv("RUN_DIR")) || tfruns::is_run_active()) &&
      !have_tensorboard_callback)
    callbacks <- append(callbacks, callback_tensorboard())

  # return the callbacks
  callbacks
}

empty_fun <- function(batch, logs = NULL) {}


