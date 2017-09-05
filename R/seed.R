
#' Set the random seed for a Keras session
#' 
#' Set various random seeds required to promote reproducible results.
#' 
#' @param seed A single value, interpreted as an integer
#' 
#' @details The provided `seed` value will establish a new random
#' seed for R, Python, NumPy, and TensorFlow (when it is used as
#' the Keras backend). 
#' 
#' @return Keras session object, invisibly
#' 
#' @export
set_keras_session_seed <- function(seed) {
  
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
  # source of non-reproducible results.k For further details, see: 
  # https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
  if (is_backend("tensorflow")) {
    
    tf <- tensorflow::tf
    session_conf = tf$ConfigProto(intra_op_parallelism_threads = 1L, 
                                  inter_op_parallelism_threads = 1L)
    
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