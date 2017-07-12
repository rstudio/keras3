
skip_if_no_keras <- function() {
  if (!have_keras())
    skip("keras not available for testing")
}

have_keras <- function() {
  implementation_module <- keras:::resolve_implementation_module()
  reticulate::py_module_available(implementation_module)
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

is_backend <- function(name) {
  identical(backend()$backend(), name)
}

define_model <- function() {
  model <- keras_model_sequential() 
  model %>%
    layer_dense(32, input_shape = 784, kernel_initializer = initializer_ones()) %>%
    layer_activation('relu') %>%
    layer_dense(10) %>%
    layer_activation('softmax')
  model
}

define_and_compile_model <- function() {
  model <- define_model()
  model %>% 
    compile(
      loss='binary_crossentropy',
      optimizer = optimizer_sgd(),
      metrics='accuracy'
    )
  model
}




