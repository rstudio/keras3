
use_virtualenv("~/tensorflow")

skip_if_no_keras <- function() {
  if (!reticulate::py_module_available("keras"))
    skip("keras not available for testing")
}

test_call_succeeds <- function(call_name, expr) {
  test_that(paste(call_name, "call succeeds"), {
    skip_if_no_keras()
    force(expr)
    expect_equal(TRUE, TRUE)
  })
}
