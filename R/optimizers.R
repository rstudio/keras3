

#' Stochastic gradient descent optimizer
#' 
#' Stochastic gradient descent optimizer with support for momentum, learning 
#' rate decay, and Nesterov momentum.
#' 
#' @param lr float >= 0. Learning rate.
#' @param momentum float >= 0. Parameter updates momentum.
#' @param decay float >= 0. Learning rate decay over each update.
#' @param nesterov boolean. Whether to apply Nesterov momentum.
#' @param clipnorm Gradients will be clipped when their L2 norm exceeds this
#'   value.
#' @param clipvalue Gradients will be clipped when their absolute value exceeds
#'   this value.
#' 
#' @return Optimizer for use with \code{\link{compile}}.
#' 
#' @export
optimizer_sgd <- function(lr = 0.01, momentum = 0.0, decay = 0.0, nesterov = FALSE,
                          clipnorm = NULL, clipvalue = NULL) {
  keras$optimizers$SGD(
    lr = lr,
    momentum = momentum,
    decay = decay,
    nesterov = nesterov,
    clipnorm = clipnorm,
    clipvalue = clipvalue
  )
}

#' RMSProp optimizer
#' 
#' @inheritParams optimizer_sgd
#' @param rho float >= 0. Decay factor.
#' @param epsilon float >= 0. Fuzz factor. 
#' 
#' @note It is recommended to leave the parameters of this optimizer at their
#' default values (except the learning rate, which can be freely tuned).
#' 
#' This optimizer is usually a good choice for recurrent neural networks.
#' 
#' @export
optimizer_rmsprop <- function(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0,
                              clipnorm = NULL, clipvalue = NULL) {
  keras$optimizers$RMSprop(
    lr = lr,
    rho = rho,
    epsilon = epsilon,
    decay = decay,
    clipnorm = clipnorm,
    clipvalue = clipvalue
  )
}


#' Adagrad optimizer.
#' 
#' Adagrad optimizer as described in
#' \href{http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf}{Adaptive 
#' Subgradient Methods for Online Learning and Stochastic Optimization}.
#' 
#' @inheritParams optimizer_rmsprop
#' 
#' @note It is recommended to leave the parameters of this optimizer at their 
#'   default values.
#' 
#' @export
optimizer_adagrad <- function(lr = 0.01, epsilon = 1e-08, decay = 0.0,
                              clipnorm = NULL, clipvalue = NULL) {
  keras$optimizers$Adagrad(
    lr = lr,
    epsilon = epsilon,
    decay = decay,
    clipnorm = clipnorm,
    clipvalue = clipvalue
  )
}

#' Adadelta optimizer.
#' 
#' Adadelta optimizer as described in
#' \href{https://arxiv.org/abs/1212.5701}{ADADELTA: An Adaptive Learning Rate
#' Method}.
#' 
#' @inheritParams optimizer_rmsprop
#'   
#' @note It is recommended to leave the parameters of this optimizer at their 
#'   default values.
#'   
#' @export
optimizer_adadelta <- function(lr = 1.0, rho = 0.95, epsilon = 1e-08, decay = 0.0,
                               clipnorm = NULL, clipvalue = NULL) {
  keras$optimizers$Adadelta(
    lr = lr,
    rho = rho,
    epsilon = epsilon,
    decay = decay,
    clipnorm = normalize_clip(clipnorm),
    clipvalue = normalize_clip(clipvalue)
  )
}

#' Adam optimizer
#' 
#' Adam optimizer as described in \href{https://arxiv.org/abs/1412.6980v8}{Adam 
#' - A Method for Stochastic Optimization}.
#' 
#' @inheritParams optimizer_rmsprop
#' @param beta_1 The exponential decay rate for the 1st moment estimates. float,
#'   0 < beta < 1. Generally close to 1.
#' @param beta_2 The exponential decay rate for the 2nd moment estimates. float,
#'   0 < beta < 1. Generally close to 1.
#'   
#' @note Default parameters follow those provided in the original paper.
#'   
#' @seealso \code{\link{optimizer_adamax}}
#'   
#' @export
optimizer_adam <- function(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0,
                           clipnorm = NULL, clipvalue = NULL) {
  keras$optimizers$Adam(
    lr = lr,
    beta_1 = beta_1,
    beta_2 = beta_2,
    epsilon = epsilon,
    decay = decay,
    clipnorm = normalize_clip(clipnorm),
    clipvalue = normalize_clip(clipvalue)
  )
}

#' Adamax optimizer
#' 
#' Adamax optimizer from Section 7 of the \href{https://arxiv.org/abs/1412.6980v8}{Adam paper}.
#' It is a variant of Adam based on the infinity norm.
#' 
#' @inheritParams optimizer_adam
#' 
#' @seealso \code{\link{optimizer_adam}}
#' 
#' @export
optimizer_adamax <- function(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0,
                             clipnorm = NULL, clipvalue = NULL) {
  keras$optimizers$Adamax(
    lr = lr,
    beta_1 = beta_1,
    beta_2 = beta_2,
    epsilon = epsilon,
    decay = decay,
    clipnorm = normalize_clip(clipnorm),
    clipvalue = normalize_clip(clipvalue)
  )
}

#' Nesterov Adam optimizer
#' 
#' Much like Adam is essentially RMSprop with momentum, Nadam is Adam RMSprop 
#' with Nesterov momentum. See 
#' \href{http://cs229.stanford.edu/proj2015/054_report.pdf}{Incorporating 
#' Nesterov Momentum into Adam}.
#' 
#' @inheritParams optimizer_adam
#' @param schedule_decay Schedule deacy.
#'   
#' @details Default parameters follow those provided in the paper. It is 
#'   recommended to leave the parameters of this optimizer at their default 
#'   values.
#'   
#' @seealso \href{http://www.cs.toronto.edu/~fritz/absps/momentum.pdf}{On the
#'   importance of initialization and momentum in deep learning}.
#'   
#' @export
optimizer_nadam <- function(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, 
                            schedule_decay = 0.004, clipnorm = NULL, clipvalue = NULL) {
  keras$optimizers$Nadam(
    lr = lr,
    beta_1 = beta_1,
    beta_2 = beta_2,
    epsilon = epsilon,
    schedule_decay = schedule_decay,
    clipnorm = normalize_clip(clipnorm),
    clipvalue = normalize_clip(clipvalue)
  )
}


# normalize NULL clip value to 0 (a no-op)
normalize_clip <- function(clip) {
  if (is.null(clip))
    0
  else
    clip
}
