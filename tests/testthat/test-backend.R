context("backend")

source("utils.R")

test_succeeds("backend returns numpy array when convert = FALSE", {
  K <- backend(convert = FALSE)
  expect_true(inherits(K$cast_to_floatx(42), "numpy.ndarray"))
})

test_backend <- function(name, expr, required_version = NULL) {
  test_succeeds(required_version = required_version,
                paste0(name, " backend function"), expr)
}

test_backend("k_abs", k_abs(42))

test_backend("k_all", {
  skip_if_cntk()
  logical_vals <- k_constant(array(TRUE, dim = c(3,5)), dtype = "bool")
  k_all(logical_vals)
  k_all(logical_vals, axis = 1)
  k_all(logical_vals, axis = 2)
})
  
test_backend("k_any", {
  skip_if_cntk()
  logical_vals <- k_constant(array(TRUE, dim = c(3,5)), dtype = "bool")
  k_any(logical_vals)
  k_any(logical_vals, axis = 1)
  k_any(logical_vals, axis = 2)
})


test_backend("k_arange", {
  skip_if_cntk()
  logical_vals <- k_constant(array(TRUE, dim = c(3,5)), dtype = "bool")
  k_arange(10)
  k_arange(1, 11)
  k_arange(1, 11, 2)
})
  
test_backend("k_clear_session", {
  if (is_backend("tensorflow"))
    k_clear_session()
})
            

test_backend("k_argmax", {
  float_vals <- k_variable(array(runif(3*5), dim = c(3,5)))
  x <- k_variable(array(runif(10*28*28), dim = c(10,28,28)))
  y <- k_variable(array(runif(10*28*28), dim = c(10,28,28)))
  k_argmax(float_vals)
  k_argmax(float_vals, axis = 1)
  k_argmax(float_vals, axis = 2)
})
test_backend("k_argmin", {
  float_vals <- k_variable(array(runif(3*5), dim = c(3,5)))
  x <- k_variable(array(runif(10*28*28), dim = c(10,28,28)))
  y <- k_variable(array(runif(10*28*28), dim = c(10,28,28)))
  k_argmin(float_vals)
  k_argmin(float_vals, axis = 1)
  k_argmin(float_vals, axis = 2)
})

test_backend("k_backend", k_backend())

test_backend("k_batch_dot", {
  k_batch_dot(x, y, axes = c(2,3))
})

test_backend("k_batch_flatten", k_batch_flatten(x))

test_backend("k_batch_get_value", k_batch_get_value(list(x,y)))

test_backend("k_batch_normalization, k_mean, k_std", {
  mean <- k_mean(x, axis = c(1,2))
  sd <- k_std(x, axis = c(1,2))
  k_batch_normalization(x, mean, sd, 
                        beta = mean, 
                        gamma = mean)
})

test_backend("k_batch_set_value", {
  var = k_variable(10)
  k_batch_set_value(list(list(var, 20)))
})

test_backend("k_bias_add", {
  k_bias_add(x, k_constant(c(1:28), dtype = "float32"))
})

test_backend("k_binary_crossentropy", {
  k_binary_crossentropy(x, y)
})

test_backend("k_cast_to_floatx", {
  k_cast_to_floatx(c(1:20))  
})

test_backend("k_cast_", {
  k_cast(x, dtype = "float64")  
})

test_backend("k_categorical_crossentropy", {
  k_categorical_crossentropy(x, y)
})

test_backend("k_clip", {
  k_clip(k_constant(c(1:10)), 5, 8)
})

test_backend("k_concatenate", {
  k_concatenate(list(x, y), 1)
})

