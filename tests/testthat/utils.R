
use_virtualenv("~/tensorflow")

skip_if_no_keras <- function() {
  if (!reticulate::py_module_available("keras"))
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
