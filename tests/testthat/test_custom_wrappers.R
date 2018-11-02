


#context("custom-wrappers")

#source("utils.R")

# Custom wrapper class
CustomWrapper <- R6::R6Class(
  
  "CustomWrapper",
  
  inherit = KerasWrapper,
  
  public = list(
    weight_shape = NULL,
    weight_init = NULL,
    custom_weight = NULL,
    
    initialize = function(weight_shape,
                          weight_init) {
      self$weight_shape <- weight_shape
      self$weight_init <- weight_init

    },
    
    build = function(input_shape) {
      
      if (!private$py_wrapper$layer$built) private$py_wrapper$layer$build(input_shape)
      
      self$custom_weight <- private$py_wrapper$layer$add_weight(
        name = 'custom_weight',
        shape = self$weight_shape,
        initializer = self$weight_init,
        trainable = TRUE
      )

      regularizer <- k_log(self$custom_weight)
      private$py_wrapper$layer$add_loss(regularizer)
      
    },

    call = function(inputs, mask = NULL, training = NULL) {
      private$py_wrapper$layer$call(inputs)
    },
    
    compute_output_shape = function(input_shape) {
      private$py_wrapper$layer$compute_output_shape(input_shape)
    }
  )
)

# wrapper instantiator
wrapper_custom <-
  function(object,
           layer,
           weight_shape,
           weight_init,
           input_shape = NULL) {
    create_wrapper(
      CustomWrapper,
      object,
      list(
        layer = layer,
        weight_shape = weight_shape,
        weight_init = weight_init,
        input_shape = input_shape
      )
    )
  }


#test_succeeds("Use an R-based custom Keras wrapper", {
    
    model <- keras_model_sequential() %>%
      wrapper_custom(
        layer = layer_dense(units = 4),
        weight_shape = shape(1),
        weight_init = initializer_he_normal(),
        input_shape = shape(2)
    ) %>%
    wrapper_custom(layer = layer_dense(units = 2),
                   weight_shape = shape(1),
                   weight_init = initializer_he_normal()
                   ) %>%
    layer_dense(units = 1)
  
  model %>% compile(optimizer = "adam", loss = "mse")
  
  model %>% fit(x = matrix(1:10, ncol = 2),
                y = matrix(1:5, ncol = 1),
                batch_size = 1,
                epochs = 1)
  m <- model$layers[[1]]
  m$get_weights()
  
#})
  
  
  