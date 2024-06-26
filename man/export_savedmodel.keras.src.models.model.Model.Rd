% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/model-persistence.R
\name{export_savedmodel.keras.src.models.model.Model}
\alias{export_savedmodel.keras.src.models.model.Model}
\title{Create a TF SavedModel artifact for inference (e.g. via TF-Serving).}
\usage{
\method{export_savedmodel}{keras.src.models.model.Model}(object, export_dir_base, ...)
}
\arguments{
\item{object}{A keras model.}

\item{export_dir_base}{string, file path where to save
the artifact.}

\item{...}{For forward/backward compatability.}
}
\value{
This is called primarily for the side effect of exporting \code{object}.
The first argument, \code{object} is also returned, invisibly, to enable usage
with the pipe.
}
\description{
(e.g. via TF-Serving).

\strong{Note:} This can currently only be used with
the TensorFlow or JAX backends.

This method lets you export a model to a lightweight SavedModel artifact
that contains the model's forward pass only (its \code{call()} method)
and can be served via e.g. TF-Serving. The forward pass is registered
under the name \code{serve()} (see example below).

The original code of the model (including any custom layers you may
have used) is \emph{no longer} necessary to reload the artifact -- it is
entirely standalone.
}
\section{Examples}{
\if{html}{\out{<div class="sourceCode r">}}\preformatted{# Create the artifact
model |> tensorflow::export_savedmodel("path/to/location")

# Later, in a different process/environment...
library(tensorflow)
reloaded_artifact <- tf$saved_model$load("path/to/location")
predictions <- reloaded_artifact$serve(input_data)

# see tfdeploy::serve_savedmodel() for serving a model over a local web api.
}\if{html}{\out{</div>}}
}

\seealso{
Other saving and loading functions: \cr
\code{\link{layer_tfsm}()} \cr
\code{\link{load_model}()} \cr
\code{\link{load_model_weights}()} \cr
\code{\link{register_keras_serializable}()} \cr
\code{\link{save_model}()} \cr
\code{\link{save_model_config}()} \cr
\code{\link{save_model_weights}()} \cr
\code{\link{with_custom_object_scope}()} \cr
}
\concept{saving and loading functions}
