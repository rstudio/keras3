context("custom-wrappers")



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

      super$build(input_shape)

      self$custom_weight <- super$add_weight(
        name = "custom_weight",
        shape = self$weight_shape,
        initializer = self$weight_init,
        trainable = TRUE
      )

      regularizer <- k_sum(k_log(self$custom_weight))
      super$add_loss(regularizer)
    },

    call = function(x, mask = NULL, training = NULL) {
      out <- super$call(x)
      k_sum(self$custom_weight) + out
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


test_succeeds("Use an R-based custom Keras wrapper", {

  model <- keras_model_sequential() %>%
    wrapper_custom(
      layer = layer_dense(units = 4),
      weight_shape = shape(1),
      weight_init = initializer_he_normal(),
      input_shape = shape(2)
    ) %>%
    wrapper_custom(
      layer = layer_dense(units = 2),
      weight_shape = shape(1),
      weight_init = initializer_he_normal()
    ) %>%
    layer_dense(units = 10, kernel_regularizer = regularizer_l1()) %>%
    layer_dense(units = 1)

  model %>% compile(optimizer = "adam", loss = "mse")

  model %>% fit(
    x = matrix(1:10, ncol = 2),
    y = matrix(1:5, ncol = 1),
    batch_size = 1,
    epochs = 1
  )

  expect_true(length(model$layers[[1]]$get_weights()) == 3)
  # seems like this is no longer garanteed. Python example:
  # https://colab.research.google.com/drive/15blNyNpK_CCR2vCsAugeivqSf0NBf4S0
  # expect_true(length(model$layers[[1]]$losses) == 1)
})


test_succeeds("Custom class inheriting keras$layers$Wrapper", {

  environment(wrapper_custom) <- local({
    create_wrapper <- create_layer
    environment()
  })

  # Custom wrapper class
  CustomWrapper <- R6::R6Class(
    "CustomWrapper",

    inherit = keras$layers$Wrapper,

    public = list(
      initialize = function(weight_shape, weight_init, ...) {
        super$initialize(...)
        self$weight_shape <- weight_shape
        self$weight_init <- weight_init
      },

      build = function(input_shape) {
        self$layer$build(input_shape)

        self$custom_weight <- self$layer$add_weight(
          name = "custom_weight",
          shape = self$weight_shape,
          initializer = self$weight_init,
          trainable = TRUE
        )

        regularizer <- k_sum(k_log(self$custom_weight))
        self$layer$add_loss(regularizer)
      },

      call = function(x, mask = NULL, training = NULL) {
        out <- self$layer$call(x)
        k_sum(self$custom_weight) + out
      }

    )
  )

  shape <- function(...)
    tensorflow::tf$TensorShape(tensorflow::shape(...)) # reticulate#1023

  model <- keras_model_sequential() %>%
    wrapper_custom(
      layer = layer_dense(units = 4),
      weight_shape = shape(1),
      weight_init = initializer_he_normal(),
      input_shape = shape(2)
    ) %>%
    wrapper_custom(
      layer = layer_dense(units = 2),
      weight_shape = shape(1),
      weight_init = initializer_he_normal()
    ) %>%
    layer_dense(units = 10, kernel_regularizer = regularizer_l1()) %>%
    layer_dense(units = 1)

  model %>% compile(optimizer = "adam", loss = "mse")

  model %>% fit(
    x = matrix(1:10, ncol = 2),
    y = matrix(1:5, ncol = 1),
    batch_size = 1,
    epochs = 1
  )

  expect_true(length(model$layers[[1]]$get_weights()) == 3)
})
