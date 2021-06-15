#' Pipe operator
#'
#' See \code{\link[magrittr]{\%>\%}} for more details.
#'
#' @name %>%
#' @rdname pipe
#' @keywords internal
#' @export
#' @import magrittr
#' @usage lhs \%>\% rhs
NULL

#' Assign values to names
#'
#' See \code{\link[zeallot]{\%<-\%}} for more details.
#'
#' @name %<-%
#' @rdname multi-assign
#' @keywords internal
#' @export
#' @import zeallot
#' @usage x \%<-\% value
NULL

#' @importFrom reticulate use_python
#' @export
reticulate::use_python

#' @importFrom reticulate use_virtualenv
#' @export
reticulate::use_virtualenv

#' @importFrom reticulate use_condaenv
#' @export
reticulate::use_condaenv

#' @importFrom reticulate array_reshape
#' @export
reticulate::array_reshape

#' @importFrom reticulate tuple
#' @export
reticulate::tuple

#' @importFrom tensorflow use_session_with_seed
#' @export
tensorflow::use_session_with_seed

#' @importFrom tensorflow tensorboard
#' @export
tensorflow::tensorboard

#' @importFrom tensorflow evaluate
#' @export
tensorflow::evaluate

#' @importFrom tensorflow export_savedmodel
#' @export
tensorflow::export_savedmodel

#' @importFrom tensorflow shape
#' @export
tensorflow::shape

#' @importFrom tfruns flags
#' @export
tfruns::flags

#' @importFrom tfruns flag_numeric
#' @export
tfruns::flag_numeric

#' @importFrom tfruns flag_integer
#' @export
tfruns::flag_integer

#' @importFrom tfruns flag_string
#' @export
tfruns::flag_string

#' @importFrom tfruns flag_boolean
#' @export
tfruns::flag_boolean

#' @importFrom tfruns run_dir
#' @export
tfruns::run_dir

#' @importFrom generics fit
#' @export
generics::fit

#' @importFrom generics compile
#' @export
generics::compile
