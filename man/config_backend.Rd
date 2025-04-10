% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/config.R
\name{config_backend}
\alias{config_backend}
\title{Publicly accessible method for determining the current backend.}
\usage{
config_backend()
}
\value{
String, the name of the backend Keras is currently using. One of
\code{"tensorflow"}, \code{"torch"}, or \code{"jax"}.
}
\description{
Publicly accessible method for determining the current backend.
}
\section{Examples}{
\if{html}{\out{<div class="sourceCode r">}}\preformatted{config_backend()
}\if{html}{\out{</div>}}

\if{html}{\out{<div class="sourceCode">}}\preformatted{## [1] "tensorflow"

}\if{html}{\out{</div>}}
}

\seealso{
\code{\link[=use_backend]{use_backend()}}

Other config backend: \cr
\code{\link{config_epsilon}()} \cr
\code{\link{config_floatx}()} \cr
\code{\link{config_image_data_format}()} \cr
\code{\link{config_set_epsilon}()} \cr
\code{\link{config_set_floatx}()} \cr
\code{\link{config_set_image_data_format}()} \cr

Other backend: \cr
\code{\link{clear_session}()} \cr
\code{\link{config_epsilon}()} \cr
\code{\link{config_floatx}()} \cr
\code{\link{config_image_data_format}()} \cr
\code{\link{config_set_epsilon}()} \cr
\code{\link{config_set_floatx}()} \cr
\code{\link{config_set_image_data_format}()} \cr

Other config: \cr
\code{\link{config_disable_flash_attention}()} \cr
\code{\link{config_disable_interactive_logging}()} \cr
\code{\link{config_disable_traceback_filtering}()} \cr
\code{\link{config_dtype_policy}()} \cr
\code{\link{config_enable_flash_attention}()} \cr
\code{\link{config_enable_interactive_logging}()} \cr
\code{\link{config_enable_traceback_filtering}()} \cr
\code{\link{config_enable_unsafe_deserialization}()} \cr
\code{\link{config_epsilon}()} \cr
\code{\link{config_floatx}()} \cr
\code{\link{config_image_data_format}()} \cr
\code{\link{config_is_interactive_logging_enabled}()} \cr
\code{\link{config_is_traceback_filtering_enabled}()} \cr
\code{\link{config_set_backend}()} \cr
\code{\link{config_set_dtype_policy}()} \cr
\code{\link{config_set_epsilon}()} \cr
\code{\link{config_set_floatx}()} \cr
\code{\link{config_set_image_data_format}()} \cr
}
\concept{backend}
\concept{config}
\concept{config backend}
