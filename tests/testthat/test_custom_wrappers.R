


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
      if (!private$py_wrapper$layer$built) private$py_wrapper$layer$build(input_shape)
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
  input <- layer_input(shape = 2)
  output <- input %>%
    wrapper_custom(layer = layer_dense(units = 32),
                   weight_shape = 1,
                   weight_init = initializer_he_normal
                   ) %>%
    wrapper_custom(layer = layer_dense(units = 32),
                   weight_shape = 1,
                   weight_init = initializer_he_normal
                   ) %>%
    layer_dense(units = 1)
  
  model %>% compile(optimizer = "adam", loss = "mse")
  
  model %>% fit(x = matrix(1:10, ncol = 2),
                y = matrix(1:5, ncol = 5),
                batch_size = 1,
                epochs = 1)
  
  
#})
  
  
  