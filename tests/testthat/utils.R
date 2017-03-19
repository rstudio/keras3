
use_virtualenv("~/tensorflow")

skip_if_no_keras <- function() {
  if (!reticulate::py_module_available("keras"))
    skip("keras not available for testing")
}
