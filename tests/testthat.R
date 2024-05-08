
# options("testthat.progress.max_fails" = 15000L)


library(testthat)

library(keras3)

if (identical(Sys.getenv("NOT_CRAN"), "true")) {
  test_check("keras3")
}
