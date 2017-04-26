


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

#' Flow Input Data
#' 
#' Generic function for flowing input data from a stream. The function invokes
#' particular methods which depend on the class of the first argument.
#' 
#' @param generator A data generator (e.g. [image_data_generator()]).
#' @param ...	Additional arguments
#'   
#' @export
flow_data <- function(generator, ...) {
  UseMethod("flow_data")
}

#' @export
flow_data.default <- function(generator, ...) {
  stop("No flow_data method defined for object of class ", paste(class(generator), sep = ","))
}

#' Flow Input Data from a Directory
#' 
#' Generic function for flowing input data from a directory The function invokes
#' particular methods which depend on the class of the first argument.
#' 
#' @param generator A data generator (e.g. [image_data_generator()]).
#' @param directory Directory path to read input data from
#' @param ...	Additional arguments
#'   
#' @export
flow_data_from_directory <- function(generator, directory, ...) {
  UseMethod("flow_data_from_directory")
}

#' @export
flow_data_from_directory.default <- function(generator, directory, ...) {
  stop("No flow_data_from_directory method defined for object of class ", paste(class(generator), sep = ","))
}


