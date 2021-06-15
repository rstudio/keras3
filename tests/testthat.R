library(testthat)

if (Sys.getenv("TENSORFLOW_EAGER") == "TRUE")
  tensorflow::tfe_enable_eager_execution()

library(keras)

if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  test_check("keras")
}
