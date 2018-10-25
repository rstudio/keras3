KerasWrapper <- R6::R6Class(
  "KerasWrapper",
  
  public = list(
    build = function(input_shape) {
      ### here a subclass will need to access the wrapped layer and build it
      ### then build itself
      ### then e.g. add a weight to the layer
      ### then e.g. add a loss to the layer
      private$py_wrapper$layer$build(input_shape)
      private$py_wrapper$build()
    },
    
    call = function(inputs, mask = NULL) {
      ### call the wrapped layer
      private$py_wrapper$layer$call(inputs, mask)
    },
    
    compute_output_shape = function(input_shape) {
      ### here we will want self.layer.compute_output_shape(input_shape)
      private$py_wrapper$layer$compute_output_shape(input_shape)
    },
    
    # Add losses to the layer
    add_loss = function(losses, inputs = NULL) {
      # I don't know if we need this, as losses should (could?) be added to the wrapped layer in build
      args <- list()
      args$losses <- losses
      args$inputs <- inputs
      do.call(private$py_wrapper$layer$add_loss, args)
    },
    
    # Adds a weight variable to the layer.
    add_weight = function(name,
                          shape,
                          dtype = NULL,
                          initializer = NULL,
                          regularizer = NULL,
                          trainable = TRUE,
                          constraint = NULL) {
      # I don't know if we want to keep this and add weights to the wrapped layer instead,
      # or if we assume that has happened in build, above
      
      args <- list()
      args$name <- name
      args$shape <- shape
      args$dtype <- dtype
      args$initializer <- initializer
      args$regularizer <- regularizer
      args$trainable <- trainable
      args$constraint <- constraint

      do.call(private$wrapper$layer$add_weight, args)
    },
    
    # back reference to python wrapper that wraps us
    .set_py_wrapper = function(py_wrapper) {
      private$py_wrapper <- py_wrapper
    },
    
    python_wrapper = function() {
      private$py_wrapper
    }
  ),
  
  active = list(
    input = function(value) {
      if (missing(value))
        return(private$py_wrapper$input)
      else
        private$py_wrapper$input <- value
    },
    output = function(value) {
      if (missing(value))
        return(private$py_wrapper$output)
      else
        private$py_wrapper$output <- value
    }
  ),
  
  private = list(py_wrapper = NULL)
)


create_wrapper <- function(wrapper_class, object, args = list()) {
  
  # remove kwargs that are null
  args$layer <- args$layer
  args$input_shape <- args$input_shape
  args$batch_input_shape <- args$batch_input_shape
  args$batch_size <- args$batch_size
  args$dtype <- args$dtype
  args$name <- args$name
  args$trainable <- args$trainable
  args$weights <- args$weights
  
  common_arg_names <- c("layer", "input_shape", "batch_input_shape", "batch_size",
                          "dtype", "name", "trainable", "weights")
  py_wrapper_args <- args[common_arg_names]
  py_wrapper_args[sapply(py_wrapper_args, is.null)] <- NULL
  for (arg in names(py_wrapper_args))
  args[[arg]] <- NULL
    
  r6_wrapper <- do.call(wrapper_class$new, args)
    
  python_path <- system.file("python", package = "keras")
  tools <- reticulate::import_from_path("kerastools", path = python_path)
  py_wrapper_args$r_build <- r6_wrapper$build
  py_wrapper_args$r_call <-  r6_wrapper$call
  py_wrapper_args$r_compute_output_shape <- r6_wrapper$compute_output_shape
  py_wrapper <- do.call(tools$wrapper$RWrapper, py_wrapper_args)
    
  r6_wrapper$.set_py_wrapper(py_wrapper)  


  if (missing(object) || is.null(object))
    r6_wrapper
  else
    invisible(compose_layer(object, r6_wrapper))
}

