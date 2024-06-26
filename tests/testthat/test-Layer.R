context("Layer")

create_custom_layer <- function() {
  Layer(
    classname = "MultiplyByX",

    initialize = function(x) {
      super()$`__init__`()
      self$x <- tensorflow::tf$constant(x)
    },

    call =  function(inputs, ...) {
      inputs * self$x
    },

    get_config = function()
      list(x = as.numeric(self$x))

  )
}

create_model_with_custom_layer <- function() {

  layer_multiply_by_x <- create_custom_layer()

  layer_multiply_by_2 <- layer_multiply_by_x(x = 2)

  input <- layer_input(shape = 1)
  output <- layer_multiply_by_2(input)

  model <- keras_model(input, output)
  model
}


test_succeeds("Can create and use a custom layer", {

  skip_if_not_tensorflow_version("2.0")

  model <- create_model_with_custom_layer()

  out <- predict(model, c(1,2,3,4,5))

  expect_equal(out, matrix(1:5, ncol = 1)*2)
  expect_equal(model$get_config()$layers[[2]]$class_name, "MultiplyByX")
})

test_succeeds("Can use custom layers in sequential models", {

  skip_if_not_tensorflow_version("2.0")

  layer_multiply_by_x <- create_custom_layer()

  model <- keras_model_sequential() %>%
    layer_multiply_by_x(2) %>%
    layer_multiply_by_x(2)

  out <- predict(model, matrix(c(1,2,3,4,5), ncol = 1))

  expect_equal(out, matrix(1:5, ncol = 1)*2*2)
})

test_succeeds("Input shape is 1-based indexed", {

  concat_layer <- Layer(
    classname = "Hello",
    initialize = function() {
      super()$`__init__`()
    },
    call = function(x, ...) {
      tensorflow::tf$concat(list(x,x), axis = 1L)
    },
    compute_output_shape = function(input_shape) {
      list(input_shape[[1]], input_shape[[2]]*2L)
    }
  )

  x <- layer_input(shape = 10)
  out <- concat_layer(x)

  expect_identical(out$shape[[2]], 20L)
})

test_succeeds("Can use self$add_weight", {


  layer_dense2 <- Layer(
    "Dense2",

    initialize = function(units) {
      super()$`__init__`()
      self$units <- as.integer(units)
    },

    build = function(input_shape) {
      self$kernel <- self$add_weight(
        name = "kernel",
        shape = list(input_shape[[2]], self$units),
        initializer = "uniform",
        trainable = TRUE
      )
    },

    call = function(x, ...) {
      tensorflow::tf$matmul(x, self$kernel)
    },

    compute_output_shape = function(input_shape) {
      list(input_shape[[1]], self$units)
    }

  )

  l <- layer_dense2(units = 10)
  input <- layer_input(shape = 10L)
  output <- l(input)

  expect_length(l$weights, 1L)
})

test_succeeds("Can inherit from an R custom layer", {


  layer_base <- Layer(
    classname = "base",
    initialize = function(x) {
      super()$`__init__`()
      self$x <- x
    },

    build = function(input_shape) {
      self$k <- 3
    },

    call = function(x) {
      x
    }
  )

  layer2 <- Layer(
    inherit = layer_base,
    classname = "b2",
    initialize = function(x) {
      super()$`__init__`(x^2)
    },
    call = function(x, ...) {
      x * self$k * self$x
    }
  )

  l <- layer2(x = 2)
  expect_equal(as.numeric(l(keras_array(1))), 12)
})


test_succeeds("create_layer_wrapper", {

  SimpleDense(keras$layers$Layer) %py_class% {
    initialize <- function(units, activation = NULL) {
      super$initialize()
      self$units <- as.integer(units)
      self$activation <- activation
    }

    build <- function(input_shape) {
      input_dim <- as.integer(input_shape) %>% tail(1)
      self$W <- self$add_weight(shape = c(input_dim, self$units),
                                initializer = "random_normal")
      self$b <- self$add_weight(shape = c(self$units),
                                initializer = "zeros")
    }

    call <- function(inputs) {
      y <- tf$matmul(inputs, self$W) + self$b
      if (!is.null(self$activation))
        y <- self$activation(y)
      y
    }
  }

  layer_simple_dense <- create_layer_wrapper(SimpleDense)

  expect_identical(formals(layer_simple_dense),
                   formals(function(object, units, activation = NULL) {}))

  model <- keras_model_sequential() %>%
    layer_simple_dense(32, activation = "relu") %>%
    layer_simple_dense(64, activation = "relu") %>%
    layer_simple_dense(32, activation = "relu") %>%
    layer_simple_dense(10, activation = "softmax")

  expect_equal(length(model$layers), 4L)
})


test_succeeds("create_layer_wrapper", {

  layer_sampler <- new_layer_class(
    classname = "Sampler",
    call = function(z_mean, z_log_var) {
      epsilon <-  random_normal(shape = shape(z_mean))
      z_mean + exp(0.5 * z_log_var) * epsilon
    }
  )

  sampler <- layer_sampler()
  z_mean <- keras_array(random_array(c(128, 2)))
  z_log_var <- keras_array(random_array(c(128, 2)))
  res <- sampler(z_mean, z_log_var)
  expect_equal(dim(res), c(128, 2))

})


test_succeeds("custom layers can accept standard layer args like input_shape", {
  # https://github.com/rstudio/keras/issues/1338
  layer_simple_dense <- new_layer_class(
    classname = "SimpleDense",

    initialize = function(units, activation = NULL, ...) {
      str(list(...))
      # browser()
      super$initialize(...)
      self$units <- as.integer(units)
      self$activation <- activation
    },

    build = function(input_shape) {
      message("build()")
      input_dim <- input_shape[length(input_shape)]
      self$W <- self$add_weight(shape = c(input_dim, self$units),
                                initializer = "random_normal")
      self$b <- self$add_weight(shape = c(self$units),
                                initializer = "zeros")
      NULL
    },

    call = function(inputs) {
      y <- tf$matmul(inputs, self$W) + self$b
      if (!is.null(activation <- self$activation))
        y <- activation(y)
      y
    }
  )

  model <- keras_model_sequential() %>%
    # layer_dense(20, input_shape = 30) %>%
    layer_simple_dense(20) %>%
    layer_dense(10)

  {
  # bug in upstream, model$input errors even w/ simple built-in layers
  skip("model.input bug upstream")
  model$compile()
  model(keras_array(array(1, c(3, 30))))
  }

  expect_identical(dim(model$input), c(NA_integer_, 30L))
  expect_true(model$built)

  res <- model(random_array(1, 30))
  expect_tensor(res, shape = c(1L, 10L))

})


test_that("calling Layer() doesn't initialize Python", {
  expect_no_error(callr::r(function() {
    Sys.setenv(CUDA_VISIBLE_DEVICES="")
    options("topLevelEnvironment" = environment())
    # pretend we're in a package

    layer_multiply_by_x <- keras3::Layer(
      classname = "MultiplyByX",

      initialize = function(x) {
        super$initialize()
        self$x <- op_convert_to_tensor(x)
      },

      call =  function(inputs, ...) {
        inputs * self$x
      },

      get_config = function() {
        list(x = as.numeric(self$x))
      }
    )

    if (reticulate:::is_python_initialized())
      stop("python initialized")

    library(keras3)
    layer <- layer_multiply_by_x(x = 2)
    input <- op_convert_to_tensor(2)
    output <- layer(input)
    stopifnot(as.array(output) == as.array(input*2))

    input <- layer_input(shape())
    output <- input %>% layer_multiply_by_x(2)
    model <- keras_model(input, output)
    x <- as.array(2)
    pred <- model %>% predict(x, verbose = 0)
    stopifnot(all.equal(pred, as.array(4)))
  }))
})
