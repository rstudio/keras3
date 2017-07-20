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


#' @importFrom reticulate use_python
#' @export
reticulate::use_python

#' @importFrom reticulate use_virtualenv
#' @export
reticulate::use_virtualenv

#' @importFrom reticulate use_condaenv
#' @export
reticulate::use_condaenv

#' @importFrom tensorflow install_tensorflow
#' @export
tensorflow::install_tensorflow

#' @importFrom tensorflow tensorboard
#' @export
tensorflow::tensorboard

#' @importFrom tensorflow use_run_dir
#' @export
tensorflow::use_run_dir

#' @importFrom tensorflow run_dir
#' @export
tensorflow::run_dir

#' @importFrom tensorflow latest_run
#' @export
tensorflow::latest_run

#' @importFrom tensorflow latest_runs
#' @export
tensorflow::latest_runs

#' @importFrom tensorflow clean_runs
#' @export
tensorflow::clean_runs

#' @importFrom tensorflow tf_config
#' @export
tensorflow::tf_config




