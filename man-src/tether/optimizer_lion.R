#' Optimizer that implements the Lion algorithm.
#'
#' @description
#' The Lion optimizer is a stochastic-gradient-descent method that uses the
#' sign operator to control the magnitude of the update, unlike other adaptive
#' optimizers such as Adam that rely on second-order moments. This make
#' Lion more memory-efficient as it only keeps track of the momentum. According
#' to the authors (see reference), its performance gain over Adam grows with
#' the batch size. Because the update of Lion is produced through the sign
#' operation, resulting in a larger norm, a suitable learning rate for Lion is
#' typically 3-10x smaller than that for AdamW. The weight decay for Lion
#' should be in turn 3-10x larger than that for AdamW to maintain a
#' similar strength (lr * wd).
#'
#' # References
#' - [Chen et al., 2023](http://arxiv.org/abs/2302.06675)
#' - [Authors' implementation](
#'     http://github.com/google/automl/tree/master/lion)
#'
#' @param learning_rate
#' A float, a
#' `keras.optimizers.schedules.LearningRateSchedule` instance, or
#' a callable that takes no arguments and returns the actual value to
#' use. The learning rate. Defaults to `0.001`.
#'
#' @param beta_1
#' A float value or a constant float tensor, or a callable
#' that takes no arguments and returns the actual value to use. The
#' rate to combine the current gradient and the 1st moment estimate.
#' Defaults to `0.9`.
#'
#' @param beta_2
#' A float value or a constant float tensor, or a callable
#' that takes no arguments and returns the actual value to use. The
#' exponential decay rate for the 1st moment estimate. Defaults to
#' `0.99`.
#'
#' @param name
#' String. The name to use
#' for momentum accumulator weights created by
#' the optimizer.
#'
#' @param weight_decay
#' Float. If set, weight decay is applied.
#'
#' @param clipnorm
#' Float. If set, the gradient of each weight is individually
#' clipped so that its norm is no higher than this value.
#'
#' @param clipvalue
#' Float. If set, the gradient of each weight is clipped to be
#' no higher than this value.
#'
#' @param global_clipnorm
#' Float. If set, the gradient of all weights is clipped
#' so that their global norm is no higher than this value.
#'
#' @param use_ema
#' Boolean, defaults to False. If True, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum
#' Float, defaults to 0.99. Only used if `use_ema=True`.
#' This is the momentum to use when computing
#' the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency
#' Int or None, defaults to None. Only used if
#' `use_ema=True`. Every `ema_overwrite_frequency` steps of iterations,
#' we overwrite the model variable by its moving average.
#' If None, the optimizer
#' does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()`
#' (which updates the model
#' variables in-place). When using the built-in `fit()` training loop,
#' this happens automatically after the last epoch,
#' and you don't need to do anything.
#'
#' @param loss_scale_factor
#' Float or `None`. If a float, the scale factor will
#' be multiplied the loss before computing gradients, and the inverse of
#' the scale factor will be multiplied by the gradients before updating
#' variables. Useful for preventing underflow during mixed precision
#' training. Alternately, `keras.optimizers.LossScaleOptimizer` will
#' automatically set a loss scale factor.
#'
#' @param ...
#' Passed on to the Python callable
#'
#' @export
#' @family optimizers
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Lion>
optimizer_lion <-
function (learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.99,
    weight_decay = NULL, clipnorm = NULL, clipvalue = NULL, global_clipnorm = NULL,
    use_ema = FALSE, ema_momentum = 0.99, ema_overwrite_frequency = NULL,
    name = "lion", ..., loss_scale_factor = NULL)
{
}
