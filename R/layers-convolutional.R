

normalize_padding <- function(padding, dims) {
  normalize_scale("padding", padding, dims)
}

normalize_cropping <- function(cropping, dims) {
  normalize_scale("cropping", cropping, dims)
}

normalize_scale <- function(name, scale, dims) {

  # validate and marshall scale argument
  throw_invalid_scale <- function() {
    stop(name, " must be a list of ", dims, " integers or list of ", dims,  " lists of 2 integers",
         call. = FALSE)
  }

  # if all of the individual items are numeric then cast to integer vector
  if (all(sapply(scale, function(x) length(x) == 1 && is.numeric(x)))) {
    as.integer(scale)
  } else if (is.list(scale)) {
    lapply(scale, function(x) {
      if (length(x) != 2)
        throw_invalid_scale()
      as.integer(x)
    })
  } else {
    throw_invalid_scale()
  }
}
