

tensorflow_on_before_use_session <- function(quiet) {
  if (is_backend("tensorflow")) {
    keras$backend$clear_session()
    TRUE
  } else {
    FALSE
  }
}

tensorflow_on_use_session <- function(sess, quiet) {
  if (is_backend("tensorflow")) {
    if (tensorflow::tf_version() < "2.0")
      keras$backend$set_session(sess)
  }
}
