% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/model-creation.R
\name{set_state_tree}
\alias{set_state_tree}
\title{Assigns values to variables of the model.}
\usage{
set_state_tree(object, state_tree)
}
\arguments{
\item{object}{A keras model.}

\item{state_tree}{A dictionary representing the state tree of the model.
The keys are the variable names, and the values are nested
dictionaries representing the variable paths and their values.}
}
\description{
This method takes a dictionary (named list) of nested variable values, which
represents the state tree of the model, and assigns them to the
corresponding variables of the model. The keys (list names) represent the
variable names (e.g., \code{'trainable_variables'}, \code{'optimizer_variables'}),
and the values are nested dictionaries containing the variable
paths and their corresponding values.
}
\seealso{
Other model functions: \cr
\code{\link{get_config}()} \cr
\code{\link{get_layer}()} \cr
\code{\link{get_state_tree}()} \cr
\code{\link{keras_model}()} \cr
\code{\link{keras_model_sequential}()} \cr
\code{\link{pop_layer}()} \cr
\code{\link{summary.keras.src.models.model.Model}()} \cr
}
\concept{model functions}
