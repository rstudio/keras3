% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/optimizers-schedules.R
\name{learning_rate_schedule_cosine_decay}
\alias{learning_rate_schedule_cosine_decay}
\title{A \code{LearningRateSchedule} that uses a cosine decay with optional warmup.}
\usage{
learning_rate_schedule_cosine_decay(
  initial_learning_rate,
  decay_steps,
  alpha = 0,
  name = "CosineDecay",
  warmup_target = NULL,
  warmup_steps = 0L
)
}
\arguments{
\item{initial_learning_rate}{A float. The initial learning rate.}

\item{decay_steps}{A int. Number of steps to decay over.}

\item{alpha}{A float. Minimum learning rate value for decay as a
fraction of \code{initial_learning_rate}.}

\item{name}{String. Optional name of the operation.  Defaults to
\code{"CosineDecay"}.}

\item{warmup_target}{A float. The target learning rate for our
warmup phase. Will cast to the \code{initial_learning_rate} datatype.
Setting to \code{NULL} will skip warmup and begins decay phase from
\code{initial_learning_rate}. Otherwise scheduler will warmup from
\code{initial_learning_rate} to \code{warmup_target}.}

\item{warmup_steps}{A int. Number of steps to warmup over.}
}
\value{
A 1-arg callable learning rate schedule that takes the current optimizer
step and outputs the decayed learning rate, a scalar tensor of the
same type as \code{initial_learning_rate}.
}
\description{
See \href{https://arxiv.org/abs/1608.03983}{Loshchilov & Hutter, ICLR2016},
SGDR: Stochastic Gradient Descent with Warm Restarts.

For the idea of a linear warmup of our learning rate,
see \href{https://arxiv.org/pdf/1706.02677}{Goyal et al.}.

When we begin training a model, we often want an initial increase in our
learning rate followed by a decay. If \code{warmup_target} is an int, this
schedule applies a linear increase per optimizer step to our learning rate
from \code{initial_learning_rate} to \code{warmup_target} for a duration of
\code{warmup_steps}. Afterwards, it applies a cosine decay function taking our
learning rate from \code{warmup_target} to \code{alpha} for a duration of
\code{decay_steps}. If \code{warmup_target} is NULL we skip warmup and our decay
will take our learning rate from \code{initial_learning_rate} to \code{alpha}.
It requires a \code{step} value to  compute the learning rate. You can
just pass a backend variable that you increment at each training step.

The schedule is a 1-arg callable that produces a warmup followed by a
decayed learning rate when passed the current optimizer step. This can be
useful for changing the learning rate value across different invocations of
optimizer functions.

Our warmup is computed as:

\if{html}{\out{<div class="sourceCode r">}}\preformatted{warmup_learning_rate <- function(step) \{
  completed_fraction <- step / warmup_steps
  total_delta <- target_warmup - initial_learning_rate
  completed_fraction * total_delta
\}
}\if{html}{\out{</div>}}

And our decay is computed as:

\if{html}{\out{<div class="sourceCode r">}}\preformatted{if (is.null(warmup_target)) \{
  initial_decay_lr <- initial_learning_rate
\} else \{
  initial_decay_lr <- warmup_target
\}

decayed_learning_rate <- function(step) \{
  step <- min(step, decay_steps)
  cosine_decay <- 0.5 * (1 + cos(pi * step / decay_steps))
  decayed <- (1 - alpha) * cosine_decay + alpha
  initial_decay_lr * decayed
\}
}\if{html}{\out{</div>}}

Example usage without warmup:

\if{html}{\out{<div class="sourceCode r">}}\preformatted{decay_steps <- 1000
initial_learning_rate <- 0.1
lr_decayed_fn <- learning_rate_schedule_cosine_decay(
    initial_learning_rate, decay_steps)
}\if{html}{\out{</div>}}

Example usage with warmup:

\if{html}{\out{<div class="sourceCode r">}}\preformatted{decay_steps <- 1000
initial_learning_rate <- 0
warmup_steps <- 1000
target_learning_rate <- 0.1
lr_warmup_decayed_fn <- learning_rate_schedule_cosine_decay(
    initial_learning_rate, decay_steps, warmup_target = target_learning_rate,
    warmup_steps = warmup_steps
)
}\if{html}{\out{</div>}}

You can pass this schedule directly into a \code{optimizer}
as the learning rate. The learning rate schedule is also serializable and
deserializable using \code{keras$optimizers$schedules$serialize} and
\code{keras$optimizers$schedules$deserialize}.
}
\seealso{
\itemize{
\item \url{https://keras.io/api/optimizers/learning_rate_schedules/cosine_decay#cosinedecay-class}
}

Other optimizer learning rate schedules: \cr
\code{\link{LearningRateSchedule}()} \cr
\code{\link{learning_rate_schedule_cosine_decay_restarts}()} \cr
\code{\link{learning_rate_schedule_exponential_decay}()} \cr
\code{\link{learning_rate_schedule_inverse_time_decay}()} \cr
\code{\link{learning_rate_schedule_piecewise_constant_decay}()} \cr
\code{\link{learning_rate_schedule_polynomial_decay}()} \cr
}
\concept{optimizer learning rate schedules}
