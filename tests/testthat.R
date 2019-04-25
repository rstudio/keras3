library(testthat)

if (Sys.getenv("TENSORFLOW_EAGER") == "TRUE")
  tensorflow::tfe_enable_eager_execution()

library(keras)

test_check("keras")
