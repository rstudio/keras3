% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/initializers.R
\name{initializer_ones}
\alias{initializer_ones}
\title{Initializer that generates tensors initialized to 1.}
\usage{
initializer_ones()
}
\value{
An \code{Initializer} instance that can be passed to layer or variable
constructors, or called directly with a \code{shape} to return a Tensor.
}
\description{
Also available via the shortcut function \code{ones}.
}
\section{Examples}{
\if{html}{\out{<div class="sourceCode r">}}\preformatted{# Standalone usage:
initializer <- initializer_ones()
values <- initializer(shape = c(2, 2))
}\if{html}{\out{</div>}}

\if{html}{\out{<div class="sourceCode r">}}\preformatted{# Usage in a Keras layer:
initializer <- initializer_ones()
layer <- layer_dense(units = 3, kernel_initializer = initializer)
}\if{html}{\out{</div>}}
}

\seealso{
\itemize{
\item \url{https://keras.io/api/layers/initializers#ones-class}
}

Other constant initializers: \cr
\code{\link{initializer_constant}()} \cr
\code{\link{initializer_identity}()} \cr
\code{\link{initializer_stft}()} \cr
\code{\link{initializer_zeros}()} \cr

Other initializers: \cr
\code{\link{initializer_constant}()} \cr
\code{\link{initializer_glorot_normal}()} \cr
\code{\link{initializer_glorot_uniform}()} \cr
\code{\link{initializer_he_normal}()} \cr
\code{\link{initializer_he_uniform}()} \cr
\code{\link{initializer_identity}()} \cr
\code{\link{initializer_lecun_normal}()} \cr
\code{\link{initializer_lecun_uniform}()} \cr
\code{\link{initializer_orthogonal}()} \cr
\code{\link{initializer_random_normal}()} \cr
\code{\link{initializer_random_uniform}()} \cr
\code{\link{initializer_stft}()} \cr
\code{\link{initializer_truncated_normal}()} \cr
\code{\link{initializer_variance_scaling}()} \cr
\code{\link{initializer_zeros}()} \cr
}
\concept{constant initializers}
\concept{initializers}
