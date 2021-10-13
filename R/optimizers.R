

#' Stochastic gradient descent optimizer
#'
#' Stochastic gradient descent optimizer with support for momentum, learning
#' rate decay, and Nesterov momentum.
#'
#' @param learning_rate float >= 0. Learning rate.
#' @param momentum float >= 0. Parameter that accelerates SGD in the relevant
#'   direction and dampens oscillations.
#' @param decay float >= 0. Learning rate decay over each update.
#' @param nesterov boolean. Whether to apply Nesterov momentum.
#' @param clipnorm Gradients will be clipped when their L2 norm exceeds this
#'   value.
#' @param clipvalue Gradients will be clipped when their absolute value exceeds
#'   this value.
#' @param ... Unused, present only for backwards compatability
#'
#' @return Optimizer for use with \code{\link{compile.keras.engine.training.Model}}.
#'
#' @family optimizers
#'
#' @export
optimizer_sgd <- function(learning_rate = 0.01, momentum = 0.0, decay = 0.0, nesterov = FALSE,
                          clipnorm = NULL, clipvalue = NULL, ...) {

  backcompat_fix_rename_lr_to_learning_rate(...)

  # compose args using list so that clipnorm and clipvalue are excluded
  # from the call when they aren't sepcified
  args <- list(
    learning_rate = learning_rate,
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
#' @param epsilon float >= 0. Fuzz factor. If `NULL`, defaults to `k_epsilon()`.
#'
#' @note It is recommended to leave the parameters of this optimizer at their
#' default values (except the learning rate, which can be freely tuned).
#'
#' This optimizer is usually a good choice for recurrent neural networks.
#'
#' @family optimizers
#'
#' @export
optimizer_rmsprop <- function(learning_rate = 0.001, rho = 0.9, epsilon = NULL, decay = 0.0,
                              clipnorm = NULL, clipvalue = NULL, ...) {

  backcompat_fix_rename_lr_to_learning_rate(...)

  # compose args using list so that clipnorm and clipvalue are excluded
  # from the call when they aren't sepcified

  args <- list(
    learning_rate = learning_rate,
    rho = rho,
    epsilon = resolve_epsilon(epsilon),
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
#' Optimization](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
#'
#' @inheritParams optimizer_rmsprop
#'
#' @note It is recommended to leave the parameters of this optimizer at their
#'   default values.
#'
#' @family optimizers
#'
#' @export
optimizer_adagrad <- function(learning_rate = 0.01, epsilon = NULL, decay = 0.0,
                              clipnorm = NULL, clipvalue = NULL, ...) {

  backcompat_fix_rename_lr_to_learning_rate(...)

  # compose args using list so that clipnorm and clipvalue are excluded
  # from the call when they aren't sepcified
  args <- list(
    learning_rate = learning_rate,
    epsilon = resolve_epsilon(epsilon),
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
optimizer_adadelta <- function(learning_rate = 1.0, rho = 0.95, epsilon = NULL, decay = 0.0,
                               clipnorm = NULL, clipvalue = NULL, ...) {

  backcompat_fix_rename_lr_to_learning_rate(...)

  # compose args using list so that clipnorm and clipvalue are excluded
  # from the call when they aren't sepcified
  args <- list(
    learning_rate = learning_rate,
    rho = rho,
    epsilon = resolve_epsilon(epsilon),
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
#' @param amsgrad Whether to apply the AMSGrad variant of this algorithm from
#'   the paper "On the Convergence of Adam and Beyond".
#'
#' @note Default parameters follow those provided in the original paper.
#'
#' @section References:
#'   - [Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)
#'   - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
#'
#' @family optimizers
#'
#' @export
optimizer_adam <- function(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = NULL, decay = 0.0,
                           amsgrad = FALSE, clipnorm = NULL, clipvalue = NULL, ...) {

    backcompat_fix_rename_lr_to_learning_rate(...)

  # compose args using list so that clipnorm and clipvalue are excluded
  # from the call when they aren't sepcified
  args <- list(
    learning_rate = learning_rate,
    beta_1 = beta_1,
    beta_2 = beta_2,
    epsilon = resolve_epsilon(epsilon),
    decay = decay
  )
  args$clipnorm <- clipnorm
  args$clipvalue <- clipvalue

  if (keras_version() >= "2.1.3")
    args$amsgrad <- amsgrad

  do.call(keras$optimizers$Adam, args)
}

# TODO: decay position moved
#   tf.keras.optimizers.Adam(
#     learning_rate=0.001,
#     beta_1=0.9,
#     beta_2=0.999,
#     epsilon=1e-07,
#     amsgrad=False,
#     name='Adam',
#     **kwargs
# )

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
optimizer_adamax <- function(learning_rate = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = NULL, decay = 0.0,
                             clipnorm = NULL, clipvalue = NULL, ...) {

  backcompat_fix_rename_lr_to_learning_rate(...)

  # compose args using list so that clipnorm and clipvalue are excluded
  # from the call when they aren't sepcified
  args <- list(
    learning_rate = learning_rate,
    beta_1 = beta_1,
    beta_2 = beta_2,
    epsilon = resolve_epsilon(epsilon),
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
#'   learning](https://www.cs.toronto.edu/~fritz/absps/momentum.pdf).
#'
#' @family optimizers
#'
#' @export
optimizer_nadam <- function(learning_rate = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = NULL,
                            schedule_decay = 0.004, clipnorm = NULL, clipvalue = NULL, ...) {

  backcompat_fix_rename_lr_to_learning_rate(...)

  # compose args using list so that clipnorm and clipvalue are excluded
  # from the call when they aren't sepcified
  args <- list(
    learning_rate = learning_rate,
    beta_1 = beta_1,
    beta_2 = beta_2,
    epsilon = resolve_epsilon(epsilon),
    schedule_decay = schedule_decay
  )
  args$clipnorm <- clipnorm
  args$clipvalue <- clipvalue

  do.call(keras$optimizers$Nadam, args)
}

resolve_epsilon <- function(epsilon) {
  if (is.null(epsilon) && keras_version() < "2.1.3")
    k_epsilon()
  else
    epsilon
}

backcompat_fix_rename_lr_to_learning_rate <- function(..., lr) {
  if (!missing(lr)) {
      warning("the `lr` argument has been renamed to `learning_rate`.")
      if (!eval.parent(quote(missing(learning_rate))))
        stop("You can't supply both `lr` and `learning_rate`")
      assign("learning_rate", lr, parent.frame())
  }
  ellipsis::check_dots_empty()
}
