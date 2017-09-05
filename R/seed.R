
#' Set the random seed for a Keras session
#'
#' Set various random seeds required to ensure reproducible results. The
#' provided `seed` value will establish a new random seed for R, Python, NumPy,
#' and TensorFlow (when it is the active Keras backend). GPU and CPU parallelism
#' will also be disabled by default.
#'
#' @param seed A single value, interpreted as an integer
#' @param disable_gpu `TRUE` to disable GPU execution (see *Parallelism* below).
#' @param disable_parallel_cpu `TRUE` to disable CPU parallelism (see
#'   *Parallelism* below).
#' @param quiet `TRUE` to suppress printing of messages.
#'
#' @details This function must be called at the very top of your script
#'  (i.e. immediately after `library(keras)`).
#'
#' @section Parallelism: By default the `set_keras_seed()` function
#'   disables GPU and CPU parallelism, since both can result in
#'   non-determinisitc execution patterns (see
#'   <https://stackoverflow.com/questions/42022950/>). You can optionally enable
#'   GPU or CPU parallelism by setting the `disable_gpu` and/or
#'   `disable_parallel_cpu` parameters to `FALSE`. Note that CPU parallelism is
#'   currently disabled only for the "tensorflow" back-end (additional
#'   investigation is required to determine how to disable CPU parallelism for
#'   other backends).
#'
#' @return Keras session object, invisibly
#'
#' @examples 
#' \dontrun{
#' library(keras)
#' set_keras_seed(42)
#' }
#'
#' @export
set_keras_seed <- function(seed, 
                           disable_gpu = TRUE,
                           disable_parallel_cpu = TRUE,
                           quiet = FALSE) {
  
  # disable CUDA if requeasted
  if (disable_gpu)
    Sys.setenv(CUDA_VISIBLE_DEVICES = "")
  
  # alias to Keras backend
  K <- backend()
  
  # python imports
  os <- import("os")
  random <- import("random")
  np <- import("numpy")
 
  # Ensure reproducibility for certain hash-based operations for Python 3
  # References: https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
  #             https://github.com/fchollet/keras/issues/2280#issuecomment-306959926
  Sys.setenv(PYTHONHASHSEED = "0")      
  os$environ[["PYTHONHASHSEED"]] <- "0"
    
  # set R, python, and NumPy random seeds
  seed <- as.integer(seed)
  set.seed(seed)
  random$seed(seed)
  np$random$seed(seed)
  
  # Force TensorFlow to use single thread as multiple threads are a potential
  # source of non-reproducible results. For further details, see: 
  # https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
  if (is_backend("tensorflow")) {
    
    # alias tensorflow module
    tf <- tensorflow::tf
    
    # disable parallelism if requested
    disabled <- character()
    config <- list()
    if (disable_gpu) {
      config$device_count <-  list(gpu = 0L)
      disabled <- c(disabled, "GPU")
    }
    if (disable_parallel_cpu) {
      config$intra_op_parallelism_threads <- 1L
      config$inter_op_parallelism_threads <- 1L
      disabled <- c(disabled, "CPU parallelism")
    }
    session_conf <- do.call(tf$ConfigProto, config)
    
    # show message
    msg <- paste("Set Keras session seed to", seed)
    if (length(disabled) > 0)
      msg <- paste0(msg, " (disabled ", paste(disabled, collapse = ", "), ")")
    if (!quiet)
      message(msg)
    
    # The below tf$set_random_seed() will make random number generation in the 
    # TensorFlow backend have a well-defined initial state. For further details, 
    # see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf$set_random_seed(seed)
    
    # create session
    sess <- tf$Session(graph = tf$get_default_graph(), config = session_conf)
   
    # set it as the keras session
    K$set_session(sess)
  }
 
  # return keras session invisibly
  invisible(K$get_session())
}