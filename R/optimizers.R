

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
#' @family optimizers  
#' 
#' @export
optimizer_sgd <- function(lr = 0.01, momentum = 0.0, decay = 0.0, nesterov = FALSE,
                          clipnorm = NULL, clipvalue = NULL) {
  
  # compose args using list so that clipnorm and clipvalue are excluded
  # from the call when they aren't sepcified
  args <- list(
    lr = lr,
    momentum = momentum,
    decay = decay,
    nesterov = nesterov
  )
  args$clipnorm <- clipnorm
  args$clipvalue <- clipvalue
  do.call(keras$optimizers$SGD, args)
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
#' @family optimizers  
#' 
#' @export
optimizer_rmsprop <- function(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0,
                              clipnorm = NULL, clipvalue = NULL) {
  # compose args using list so that clipnorm and clipvalue are excluded
  # from the call when they aren't sepcified
  args <- list(
    lr = lr,
    rho = rho,
    epsilon = epsilon,
    decay = decay
  )
  args$clipnorm <- clipnorm
  args$clipvalue <- clipvalue
  do.call(keras$optimizers$RMSprop, args)
}


#' Adagrad optimizer.
#'
#' Adagrad optimizer as described in [Adaptive Subgradient Methods for Online
#' Learning and Stochastic
#' Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
#'
#' @inheritParams optimizer_rmsprop
#'
#' @note It is recommended to leave the parameters of this optimizer at their
#'   default values.
#'
#' @family optimizers
#'
#' @export
optimizer_adagrad <- function(lr = 0.01, epsilon = 1e-08, decay = 0.0,
                              clipnorm = NULL, clipvalue = NULL) {
  # compose args using list so that clipnorm and clipvalue are excluded
  # from the call when they aren't sepcified
  args <- list(
    lr = lr,
    epsilon = epsilon,
    decay = decay
  )
  args$clipnorm <- clipnorm
  args$clipvalue <- clipvalue
  do.call(keras$optimizers$Adagrad, args)
}

#' Adadelta optimizer.
#'
#' Adadelta optimizer as described in [ADADELTA: An Adaptive Learning Rate
#' Method](https://arxiv.org/abs/1212.5701).
#'
#' @inheritParams optimizer_rmsprop
#'
#' @note It is recommended to leave the parameters of this optimizer at their
#'   default values.
#'
#' @family optimizers
#'
#' @export
optimizer_adadelta <- function(lr = 1.0, rho = 0.95, epsilon = 1e-08, decay = 0.0,
                               clipnorm = NULL, clipvalue = NULL) {
  # compose args using list so that clipnorm and clipvalue are excluded
  # from the call when they aren't sepcified
  args <- list(
    lr = lr,
    rho = rho,
    epsilon = epsilon,
    decay = decay
  )
  args$clipnorm <- clipnorm
  args$clipvalue <- clipvalue
  do.call(keras$optimizers$Adadelta, args)
}

#' Adam optimizer
#'
#' Adam optimizer as described in [Adam - A Method for Stochastic
#' Optimization](https://arxiv.org/abs/1412.6980v8).
#'
#' @inheritParams optimizer_rmsprop
#' @param beta_1 The exponential decay rate for the 1st moment estimates. float,
#'   0 < beta < 1. Generally close to 1.
#' @param beta_2 The exponential decay rate for the 2nd moment estimates. float,
#'   0 < beta < 1. Generally close to 1.
#'
#' @note Default parameters follow those provided in the original paper.
#'
#' @family optimizers
#'
#' @export
optimizer_adam <- function(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0,
                           clipnorm = NULL, clipvalue = NULL) {
  # compose args using list so that clipnorm and clipvalue are excluded
  # from the call when they aren't sepcified
  args <- list(
    lr = lr,
    beta_1 = beta_1,
    beta_2 = beta_2,
    epsilon = epsilon,
    decay = decay
  )
  args$clipnorm <- clipnorm
  args$clipvalue <- clipvalue
  do.call(keras$optimizers$Adam, args)
}

#' Adamax optimizer
#' 
#' Adamax optimizer from Section 7 of the [Adam paper](https://arxiv.org/abs/1412.6980v8).
#' It is a variant of Adam based on the infinity norm.
#' 
#' @inheritParams optimizer_adam
#' 
#' @family optimizers  
#' 
#' @export
optimizer_adamax <- function(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0,
                             clipnorm = NULL, clipvalue = NULL) {
  # compose args using list so that clipnorm and clipvalue are excluded
  # from the call when they aren't sepcified
  args <- list(
    lr = lr,
    beta_1 = beta_1,
    beta_2 = beta_2,
    epsilon = epsilon,
    decay = decay
  )
  args$clipnorm <- clipnorm
  args$clipvalue <- clipvalue
  do.call(keras$optimizers$Adamax, args)
}

#' Nesterov Adam optimizer
#'
#' Much like Adam is essentially RMSprop with momentum, Nadam is Adam RMSprop
#' with Nesterov momentum.
#'
#' @inheritParams optimizer_adam
#' @param schedule_decay Schedule deacy.
#'
#' @details Default parameters follow those provided in the paper. It is
#'   recommended to leave the parameters of this optimizer at their default
#'   values.
#'
#' @seealso [On the importance of initialization and momentum in deep
#'   learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf).
#'
#' @family optimizers
#'
#' @export
optimizer_nadam <- function(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, 
                            schedule_decay = 0.004, clipnorm = NULL, clipvalue = NULL) {
  # compose args using list so that clipnorm and clipvalue are excluded
  # from the call when they aren't sepcified
  args <- list(
    lr = lr,
    beta_1 = beta_1,
    beta_2 = beta_2,
    epsilon = epsilon,
    schedule_decay = schedule_decay
  )
  args$clipnorm <- clipnorm
  args$clipvalue <- clipvalue
  do.call(keras$optimizers$Nadam, args)
}

