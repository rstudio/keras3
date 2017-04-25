


#' Model Fitting
#' 
#' Generic function for fitting models. The function invokes particular methods
#' which depend on the class of the first argument.
#' 
#' @param object An object for which fitting is desired.
#' @param ...	Additional arguments affecting the fitting
#'
#' @export
fit <- function(object, ...) {
  UseMethod("fit")
}


#' @export
fit.default <- function(object, ...) {
  stop("No fit method defined for object of class ", paste(class(object), sep = ","))
}

