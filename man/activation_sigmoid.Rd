% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/activations.R
\name{activation_sigmoid}
\alias{activation_sigmoid}
\title{Sigmoid activation function.}
\usage{
activation_sigmoid(x)
}
\arguments{
\item{x}{Input tensor.}
}
\value{
A tensor, the result from applying the activation to the input tensor \code{x}.
}
\description{
It is defined as: \code{sigmoid(x) = 1 / (1 + exp(-x))}.

For small values (<-5),
\code{sigmoid} returns a value close to zero, and for large values (>5)
the result of the function gets close to 1.

Sigmoid is equivalent to a 2-element softmax, where the second element is
assumed to be zero. The sigmoid function always returns a value between
0 and 1.
}
\seealso{
\itemize{
\item \url{https://keras.io/api/layers/activations#sigmoid-function}
}

Other activations: \cr
\code{\link{activation_celu}()} \cr
\code{\link{activation_elu}()} \cr
\code{\link{activation_exponential}()} \cr
\code{\link{activation_gelu}()} \cr
\code{\link{activation_glu}()} \cr
\code{\link{activation_hard_shrink}()} \cr
\code{\link{activation_hard_sigmoid}()} \cr
\code{\link{activation_hard_tanh}()} \cr
\code{\link{activation_leaky_relu}()} \cr
\code{\link{activation_linear}()} \cr
\code{\link{activation_log_sigmoid}()} \cr
\code{\link{activation_log_softmax}()} \cr
\code{\link{activation_mish}()} \cr
\code{\link{activation_relu}()} \cr
\code{\link{activation_relu6}()} \cr
\code{\link{activation_selu}()} \cr
\code{\link{activation_silu}()} \cr
\code{\link{activation_soft_shrink}()} \cr
\code{\link{activation_softmax}()} \cr
\code{\link{activation_softplus}()} \cr
\code{\link{activation_softsign}()} \cr
\code{\link{activation_sparse_plus}()} \cr
\code{\link{activation_sparsemax}()} \cr
\code{\link{activation_squareplus}()} \cr
\code{\link{activation_tanh}()} \cr
\code{\link{activation_tanh_shrink}()} \cr
\code{\link{activation_threshold}()} \cr
}
\concept{activations}
