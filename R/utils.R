
is_null_xptr <- function(xptr) {
  saved_attributes <- attributes(xptr)
  attributes(xptr) <- NULL
  is_null <- identical(xptr, new("externalptr"))
  attributes(xptr) <- saved_attributes
  is_null
}