#' Pipe operator
#'
#' See \code{\link[magrittr]{\%>\%}} for more details.
#'
#' @name %>%
#' @rdname pipe
#' @keywords internal
#' @returns Most commonly, the result of calling the right hand side with the
#'   left hand side as an argument: `rhs(lhs)`. See the magritter vignette for
#'   other, more advanced, usages.
#' @export
#' @export
#' @importFrom magrittr %<>% %>%
#' @usage lhs \%>\% rhs
NULL


#' @export
magrittr::`%<>%`

#' Assign values to names
#'
#' See \code{\link[zeallot]{\%<-\%}} for more details.
#'
#' @name %<-%
#' @rdname multi-assign
#' @keywords internal
#' @returns The right-hand-side argument, `value`, invisibly. This called
#'   primarily for it's side-effect of assigning symbols in the current frame.
#' @export
#' @importFrom zeallot %<-%
#' @usage x \%<-\% value
NULL

#' @importFrom reticulate use_python
#' @export
reticulate::use_python

#' @importFrom reticulate use_virtualenv
#' @export
reticulate::use_virtualenv

#' @importFrom reticulate array_reshape
#' @export
reticulate::array_reshape

#' @importFrom reticulate np_array
#' @export
reticulate::np_array

#' @importFrom reticulate tuple
#' @export
reticulate::tuple

#' @export
reticulate::iter_next

#' @export
reticulate::iterate

#' @export
reticulate::as_iterator

#' @importFrom tensorflow tensorboard
#' @export
tensorflow::tensorboard

#' @importFrom tensorflow export_savedmodel
#' @export
tensorflow::export_savedmodel

#' @importFrom tensorflow as_tensor
#' @export
tensorflow::as_tensor

#' @importFrom tensorflow all_dims
#' @export
tensorflow::all_dims

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

# ' @importFrom generics evaluate
# ' @export
# generics::evaluate
## generics::evaluate() has a different signature from tensorflow::evaluate()
## evaluate(x, ...) vs evaluate(object, ...)
## We obviously can't dispatch on `x` in the evaluate() method keras uses, since
## thats a named argument for the dataset. Meaning we can't use
## generics::evaluate(). To drop the tensorflow dep, Seems like we'll have to
## eventually export a `keras3::evaluate()` generic.

#' @importFrom tensorflow evaluate
#' @export
tensorflow::evaluate
