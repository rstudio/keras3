
use_virtualenv("~/tensorflow")

skip_if_no_keras <- function() {
  if (!reticulate::py_module_available("tensorflow.contrib.keras"))
    skip("keras not available for testing")
}


test_succeeds <- function(desc, expr) {
  test_that(desc, {
    skip_if_no_keras()
    expect_error(force(expr), NA)
  })
}

test_call_succeeds <- function(call_name, expr) {
  test_succeeds(paste(call_name, "call succeeds"), expr)
}

define_model <- function() {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = 784, kernel_initializer = initializer_ones()) %>%
    layer_activation('relu') %>%
    layer_dense(10) %>%
    layer_activation('softmax')
}

define_and_compile_model <- function() {
  define_model() %>% 
    compile(
      loss='binary_crossentropy',
      optimizer = optimizer_sgd(),
      metrics='accuracy'
    )
}
