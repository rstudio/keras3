


#context("custom-wrappers")

#source("utils.R")

# Custom wrapper class
CustomWrapper <- R6::R6Class(
  "CustomWrapper",
  
  inherit = KerasWrapper,
  
  public = list(
    weight_shape = NULL,
    weight_init = NULL,
    
    initialize = function(weight_shape,
                          weight_init) {
      self$weight_shape <- weight_shape
      self$weight_init <- weight_init
    },
    
    build = function(input_shape) {
      private$py_wrapper$layer$build(input_shape)
      private$py_wrapper$build()
      private$wrapper$layer$add_weight(
        name = 'custom_weight',
        shape = weight_shape,
        initializer = weight_init,
        trainable = TRUE
      )
    },
    
    call = function(x, mask = NULL) {
      private$py_wrapper$layer$call(inputs, mask)
    },
    
    compute_output_shape = function(input_shape) {
      private$py_wrapper$layer$compute_output_shape(input_shape)
    }
  )
)

# create layer wrapper function
wrapper_custom <-
  function(object,
           layer,
           weight_shape,
           weight_init) {
    create_wrapper(
      CustomWrapper,
      object,
      list(
        layer = layer,
        weight_shape = weight_shape,
        weight_init = weight_init
      )
    )
  }


#test_succeeds("Use an R-based custom Keras wrapper", {
  model <- keras_model_sequential()
  model %>%
    wrapper_custom(layer = layer_dense(units = 32, input_shape = c(32, 32)),
                   weight_shape = 1,
                   weight_init = initializer_he_normal()) %>%
    layer_dense(units = 1)
#})