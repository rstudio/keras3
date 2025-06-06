% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/model-creation.R
\name{pop_layer}
\alias{pop_layer}
\title{Remove the last layer in a Sequential model}
\usage{
pop_layer(object, rebuild = TRUE)
}
\arguments{
\item{object}{Sequential keras model object}

\item{rebuild}{\code{bool}. Whether to rebuild the model after removing
the layer. Defaults to \code{TRUE}.}
}
\value{
The removed layer.
}
\description{
Remove the last layer in a Sequential model
}
\seealso{
Other model functions: \cr
\code{\link{get_config}()} \cr
\code{\link{get_layer}()} \cr
\code{\link{get_state_tree}()} \cr
\code{\link{keras_model}()} \cr
\code{\link{keras_model_sequential}()} \cr
\code{\link{set_state_tree}()} \cr
\code{\link{summary.keras.src.models.model.Model}()} \cr
}
\concept{model functions}
