% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/random.R
\name{random_integer}
\alias{random_integer}
\title{Draw random integers from a uniform distribution.}
\usage{
random_integer(shape, minval, maxval, dtype = "int32", seed = NULL)
}
\arguments{
\item{shape}{The shape of the random values to generate.}

\item{minval}{integer, lower bound of the range of
random values to generate (inclusive).}

\item{maxval}{integer, upper bound of the range of
random values to generate (exclusive).}

\item{dtype}{Optional dtype of the tensor. Only integer types are
supported. If not specified, \code{"int32"} is used.}

\item{seed}{Optional R integer or instance of
\code{\link[=random_seed_generator]{random_seed_generator()}}.
By default, the \code{seed} argument is \code{NULL}, and an internal global
\code{\link[=random_seed_generator]{random_seed_generator()}} is used. The \code{seed} argument can be
used to ensure deterministic (repeatable) random number generation.
Note that passing an integer as the \code{seed} value will produce the
same random values for each call. To generate different random
values for repeated calls, an instance of
\code{\link[=random_seed_generator]{random_seed_generator()}} must be provided as the \code{seed} value.

Remark concerning the JAX backend: When tracing functions with the
JAX backend the global \code{\link[=random_seed_generator]{random_seed_generator()}} is not
supported. Therefore, during tracing the default value \code{seed=NULL}
will produce an error, and a \code{seed} argument must be provided.}
}
\value{
A tensor of random values.
}
\description{
The generated values follow a uniform distribution in the range
\verb{[minval, maxval)}. The lower bound \code{minval} is included in the range,
while the upper bound \code{maxval} is excluded.

\code{dtype} must be an integer type.
}
\seealso{
Other random: \cr
\code{\link{random_beta}()} \cr
\code{\link{random_binomial}()} \cr
\code{\link{random_categorical}()} \cr
\code{\link{random_dropout}()} \cr
\code{\link{random_gamma}()} \cr
\code{\link{random_normal}()} \cr
\code{\link{random_seed_generator}()} \cr
\code{\link{random_shuffle}()} \cr
\code{\link{random_truncated_normal}()} \cr
\code{\link{random_uniform}()} \cr
}
\concept{random}
