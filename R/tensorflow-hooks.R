

tensorflow_on_before_use_session <- function(quiet) {
  if (identical(config_backend(), "tensorflow")) {
    tryCatch(
      keras$utils$clear_session(),
      python.builtin.AttributeError = function(e) {
        tryCatch(
          keras$backend$clear_session(),
          error = function(e2)
            stop(e)
        )
      }
    )
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
