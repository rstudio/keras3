
skip_if_no_keras <- function(required_version = NULL) {
  if (!have_keras(required_version))
    skip("required keras version not available for testing")
}

have_keras <- function(required_version = NULL) {
  implementation_module <- keras:::resolve_implementation_module()
  if (reticulate::py_module_available(implementation_module)) {
    if (!is.null(required_version))
      keras:::keras_version() >= required_version
    else
      TRUE
  } else {
    FALSE
  }
}


test_succeeds <- function(desc, expr, required_version = NULL) {
  test_that(desc, {
    skip_if_no_keras(required_version)
    expect_error(force(expr), NA)
  })
}

test_call_succeeds <- function(call_name, expr, required_version = NULL) {
  test_succeeds(paste(call_name, "call succeeds"), expr, required_version)
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




