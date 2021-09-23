context("custom-layers")



# Custom layer class
CustomLayer <- R6::R6Class("CustomLayer",

  inherit = KerasLayer,

  public = list(

    output_dim = NULL,

    kernel = NULL,

    initialize = function(output_dim) {
      self$output_dim <- output_dim
    },

    build = function(input_shape) {
      self$kernel <- self$add_weight(
        name = 'kernel',
        shape = list(input_shape[[2]], self$output_dim),
        initializer = initializer_random_normal(),
        trainable = TRUE
      )
    },

    call = function(x, mask = NULL) {
      self$add_loss(list(k_constant(5)))
      k_dot(x, self$kernel)
    },

    compute_output_shape = function(input_shape) {
      list(input_shape[[1]], self$output_dim)
    }
  )
)

# create layer wrapper function
layer_custom <- function(object, output_dim, name = NULL, trainable = TRUE) {
  create_layer(CustomLayer, object, list(
    output_dim = as.integer(output_dim),
    name = name,
    trainable = trainable
  ))
}


test_succeeds("Use an R-based custom Keras layer", {

  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 32, input_shape = c(32,32)) %>%
    layer_custom(output_dim = 32)
})

test_succeeds("Custom layer with time distributed layer", {

  CustomLayer <- R6::R6Class(
    "CustomLayer",

    inherit = KerasLayer,

    public = list(

      output_dim = NULL,

      kernel = NULL,

      initialize = function(output_dim) {
        self$output_dim <- as.integer(output_dim)
      },

      build = function(input_shape) {
        self$kernel <- self$add_weight(
          name = 'kernel',
          shape = list(input_shape[[2]], self$output_dim),
          initializer = initializer_random_normal(),
          trainable = TRUE
        )
      },

      call = function(x, mask = NULL) {
        k_dot(x, self$kernel)
      },

      compute_output_shape = function(input_shape) {
        list(input_shape[[1]], self$output_dim)
      }
    )
  )

  layer_custom <- function(object, output_dim, name = NULL, trainable = TRUE) {
    create_layer(CustomLayer, object, list(
      output_dim = as.integer(output_dim),
      name = name,
      trainable = trainable
    ))
  }

  x <- array(1, dim = c(100, 4, 4, 4))
  td <- time_distributed(layer = layer_custom(output_dim = 32))
  o <- td(x)

  expect_equal(o$shape$as_list(), c(100, 4,4,32))

})


test_succeeds("R6 Custom layers can inherit from a python type", {

  CustomLayer <- R6::R6Class(
    "CustomLayer",

    inherit = keras$layers$Layer,

    public = list(
      output_dim = NULL,
      kernel = NULL,

      initialize = function(output_dim, ...) {
        super()$"__init__"(...)
        self$output_dim <- output_dim
      },

      build = function(input_shape) {
        self$kernel <- self$add_weight(
          name = 'kernel',
          shape = list(input_shape[[2]], self$output_dim),
          initializer = initializer_random_normal(),
          trainable = TRUE
        )
      },

      call = function(x, mask = NULL) {
        self$add_loss(list(k_constant(5)))
        k_dot(x, self$kernel)
      },

      compute_output_shape = function(input_shape) {
        list(input_shape[[1]], self$output_dim)
      }
    )
  )


  layer_custom <- function(object, output_dim, name = NULL, trainable = TRUE) {
    create_layer(CustomLayer, object, list(
      output_dim = as.integer(output_dim),
      name = name,
      trainable = trainable
    ))
  }


  model <- keras_model_sequential()
  model %>%
    layer_dense(units = 32, input_shape = c(32,32)) %>%
    layer_custom(output_dim = 32)

  expect_tensor(model(random_array(c(3, 32, 32))))

  # can instantiate and use like a conventional layer too
  input <- layer_input(shape(1))
  expect_tensor(keras$layers$Dense(units = 32)(input),
                shape = list(NULL, 32L))

  expect_tensor(r_to_py(CustomLayer, convert = TRUE)(output_dim = 32L)(input),
                shape = list(NULL, 32L))

})




test_succeeds("Custom layers can pass along masks", {
  # issue #1225
  skip_if_not_tensorflow_version("2.3")

  MyDenseLayer <- R6::R6Class(
    "CustomLayer",
    inherit = keras$layers$Layer,

    public = list(
      num_outputs = NULL,
      kernel = NULL,
      supports_masking = TRUE,

      initialize = function(num_outputs, ...) {
        super$initialize(...)
        self$num_outputs <- num_outputs
      },

      build = function(input_shape) {
        self$kernel <- self$add_weight(
          name = 'kernel', shape = list(input_shape[[2]], self$num_outputs))
      },

      call = function(x, mask = NULL) { mask },

      compute_mask = function(x, mask = NULL) { mask },

      compute_output_shape = function(input_shape) {
        list(input_shape[[1]], self$num_outputs)
      }
    )
  )

  layer_my_dense <- function(object, num_outputs, name = NULL, trainable = TRUE) {
      create_layer(MyDenseLayer, object, list(
          num_outputs = as.integer(num_outputs),
          name = name,
          trainable = trainable
      ))
  }

  inputs <- keras$layers$Input(shape=list(10L))
  maskingLayer = keras$layers$Masking(mask_value=-9)
  masked_input = maskingLayer(inputs)
  custom_layer = layer_my_dense(num_outputs=5L)
  custom_layer_output = custom_layer(masked_input)

  expect_true(custom_layer$supports_masking)
  expect_tensor(custom_layer$input_mask)
  expect_tensor(custom_layer$output_mask, shaped_as = custom_layer$input_mask)
})
