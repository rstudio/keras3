context("backend")

source("utils.R")

test_succeeds("backend returns numpy array when convert = FALSE", {
  K <- backend(convert = FALSE)
  expect_true(inherits(K$cast_to_floatx(42), "numpy.ndarray"))
})

