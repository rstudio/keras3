
# options("testthat.progress.max_fails" = 15000L)


library(testthat)

if (Sys.getenv("TENSORFLOW_EAGER") == "TRUE")
  tensorflow::tfe_enable_eager_execution()

library(keras3)

if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  test_check("keras")
}
