

#' Stochastic gradient descent optimizer
#' 
#' Stochastic gradient descent optimizer with support for momentum, learning 
#' rate decay, and Nesterov momentum.
#' 
#' @param lr Learning rate.
#' @param momentum Parameter updates momentum.
#' @param decay Learning rate decay over each update.
#' @param nesterov Whether to apply Nesterov momentum.
#' @param clipnorm Gradients will be clipped when their L2 norm exceeds this
#'   value.
#' @param clipvalue Gradients will be clipped when their absolute value exceeds
#'   this value.
#' 
#' @return Optimizer for use with \code{\link{model_compile}}.
#' 
#' @export
optimizer_sgd <- function(lr = 0.01, momentum = 0.0, decay = 0.0, nesterov = FALSE,
                          clipnorm = NULL, clipvalue = NULL) {
  keras$optimizers$SGD(
    lr = lr,
    momentum = momentum,
    decay = decay,
    nesterov = nesterov,
    clipnorm = normalize_clip(clipnorm),
    clipvalue = normalize_clip(clipvalue)
  )
}

#' RMSProp optimizer
#' 
#' @inheritParams optimizer_sgd
#' @param rho rho Decay factor.
#' @param episilon Fuzz factor. 
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
