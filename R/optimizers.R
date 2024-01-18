

#' Optimizer that implements the Adadelta algorithm.
#'
#' @description
#' Adadelta optimization is a stochastic gradient descent method that is based
#' on adaptive learning rate per dimension to address two drawbacks:
#'
#' - The continual decay of learning rates throughout training.
#' - The need for a manually selected global learning rate.
#'
#' Adadelta is a more robust extension of Adagrad that adapts learning rates
#' based on a moving window of gradient updates, instead of accumulating all
#' past gradients. This way, Adadelta continues learning even when many updates
#' have been done. Compared to Adagrad, in the original version of Adadelta you
#' don't have to set an initial learning rate. In this version, the initial
#' learning rate can be set, as in most other Keras optimizers.
#'
#' # Reference
#' - [Zeiler, 2012](https://arxiv.org/abs/1212.5701)
#'
#' @param learning_rate
#' A float, a
#' [`LearningRateSchedule()]` instance, or
#' a callable that takes no arguments and returns the actual value to
#' use. The learning rate. Defaults to `0.001`. Note that `Adadelta`
#' tends to benefit from higher initial learning rate values compared
#' to other optimizers. To match the exact form in the original paper,
#' use 1.0.
#'
#' @param rho
#' A floating point value. The decay rate. Defaults to `0.95`.
#'
#' @param epsilon
#' Small floating point value for maintaining numerical stability.
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
#' Boolean, defaults to `FALSE`. If `TRUE`, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum
#' Float, defaults to 0.99. Only used if `use_ema=TRUE`.
#' This is the momentum to use when computing
#' the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency
#' Int or `NULL`, defaults to `NULL`. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations,
#' we overwrite the model variable by its moving average.
#' If `NULL`, the optimizer
#' does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()`
#' (which updates the model
#' variables in-place). When using the built-in `fit()` training loop,
#' this happens automatically after the last epoch,
#' and you don't need to do anything.
#'
#' @param loss_scale_factor
#' Float or `NULL`. If a float, the scale factor will
#' be multiplied the loss before computing gradients, and the inverse of
#' the scale factor will be multiplied by the gradients before updating
#' variables. Useful for preventing underflow during mixed precision
#' training. Alternately, [`optimizer_loss_scale()`] will
#' automatically set a loss scale factor.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family optimizers
#' @seealso
#' + <https://keras.io/api/optimizers/adadelta#adadelta-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adadelta>
#' @tether keras.optimizers.Adadelta
optimizer_adadelta <-
function (learning_rate = 0.001, rho = 0.95, epsilon = 1e-07,
    weight_decay = NULL, clipnorm = NULL, clipvalue = NULL, global_clipnorm = NULL,
    use_ema = FALSE, ema_momentum = 0.99, ema_overwrite_frequency = NULL,
    name = "adadelta", ..., loss_scale_factor = NULL)
{
    args <- capture_args(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$Adadelta, args)
}


#' Optimizer that implements the Adafactor algorithm.
#'
#' @description
#' Adafactor is commonly used in NLP tasks, and has the advantage
#' of taking less memory because it only saves partial information of previous
#' gradients.
#'
#' The default argument setup is based on the original paper (see reference).
#' When gradients are of dimension > 2, Adafactor optimizer will delete the
#' last 2 dimensions separately in its accumulator variables.
#'
#' # Reference
#' - [Shazeer, Noam et al., 2018](https://arxiv.org/abs/1804.04235).
#'
#' @param learning_rate
#' A float, a
#' [`LearningRateSchedule()`] instance, or
#' a callable that takes no arguments and returns the actual value to
#' use. The learning rate. Defaults to `0.001`.
#'
#' @param beta_2_decay
#' float, defaults to -0.8. The decay rate of `beta_2`.
#'
#' @param epsilon_1
#' float, defaults to 1e-30. A small offset to keep denominator
#' away from 0.
#'
#' @param epsilon_2
#' float, defaults to 1e-3. A small offset to avoid learning
#' rate becoming too small by time.
#'
#' @param clip_threshold
#' float, defaults to 1.0. Clipping threshold. This is a
#' part of Adafactor algorithm, independent from `clipnorm`,
#' `clipvalue`, and `global_clipnorm`.
#'
#' @param relative_step
#' bool, defaults to `TRUE`. If `learning_rate` is a
#' constant and `relative_step=TRUE`, learning rate will be adjusted
#' based on current iterations. This is a default learning rate decay
#' in Adafactor.
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
#' Boolean, defaults to `FALSE`. If `TRUE`, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum
#' Float, defaults to 0.99. Only used if `use_ema=TRUE`.
#' This is the momentum to use when computing
#' the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency
#' Int or `NULL`, defaults to `NULL`. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations,
#' we overwrite the model variable by its moving average.
#' If `NULL`, the optimizer
#' does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()`
#' (which updates the model
#' variables in-place). When using the built-in `fit()` training loop,
#' this happens automatically after the last epoch,
#' and you don't need to do anything.
#'
#' @param loss_scale_factor
#' Float or `NULL`. If a float, the scale factor will
#' be multiplied the loss before computing gradients, and the inverse of
#' the scale factor will be multiplied by the gradients before updating
#' variables. Useful for preventing underflow during mixed precision
#' training. Alternately, [`optimizer_loss_scale()`] will
#' automatically set a loss scale factor.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family optimizers
#' @seealso
#' + <https://keras.io/api/optimizers/adafactor#adafactor-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adafactor>
#' @tether keras.optimizers.Adafactor
optimizer_adafactor <-
function (learning_rate = 0.001, beta_2_decay = -0.8, epsilon_1 = 1e-30,
    epsilon_2 = 0.001, clip_threshold = 1, relative_step = TRUE,
    weight_decay = NULL, clipnorm = NULL, clipvalue = NULL, global_clipnorm = NULL,
    use_ema = FALSE, ema_momentum = 0.99, ema_overwrite_frequency = NULL,
    name = "adafactor", ..., loss_scale_factor = NULL)
{
    args <- capture_args(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$Adafactor, args)
}


#' Optimizer that implements the Adagrad algorithm.
#'
#' @description
#' Adagrad is an optimizer with parameter-specific learning rates,
#' which are adapted relative to how frequently a parameter gets
#' updated during training. The more updates a parameter receives,
#' the smaller the updates.
#'
#' # Reference
#' - [Duchi et al., 2011](
#'     https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf).
#'
#' @param learning_rate
#' A float, a
#' [`LearningRateSchedule()`] instance, or
#' a callable that takes no arguments and returns the actual value to
#' use. The learning rate. Defaults to `0.001`. Note that `Adagrad`
#' tends to benefit from higher initial learning rate values compared
#' to other optimizers. To match the exact form in the original paper,
#' use `1.0`.
#'
#' @param initial_accumulator_value
#' Floating point value. Starting value for the
#' accumulators (per-parameter momentum values). Must be non-negative.
#'
#' @param epsilon
#' Small floating point value for maintaining numerical stability.
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
#' Boolean, defaults to `FALSE`. If `TRUE`, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum
#' Float, defaults to 0.99. Only used if `use_ema=TRUE`.
#' This is the momentum to use when computing
#' the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency
#' Int or `NULL`, defaults to `NULL`. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations,
#' we overwrite the model variable by its moving average.
#' If `NULL`, the optimizer
#' does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()`
#' (which updates the model
#' variables in-place). When using the built-in `fit()` training loop,
#' this happens automatically after the last epoch,
#' and you don't need to do anything.
#'
#' @param loss_scale_factor
#' Float or `NULL`. If a float, the scale factor will
#' be multiplied the loss before computing gradients, and the inverse of
#' the scale factor will be multiplied by the gradients before updating
#' variables. Useful for preventing underflow during mixed precision
#' training. Alternately, [`optimizer_loss_scale()`] will
#' automatically set a loss scale factor.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family optimizers
#' @seealso
#' + <https://keras.io/api/optimizers/adagrad#adagrad-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adagrad>
#' @tether keras.optimizers.Adagrad
optimizer_adagrad <-
function (learning_rate = 0.001, initial_accumulator_value = 0.1,
    epsilon = 1e-07, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL,
    global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99,
    ema_overwrite_frequency = NULL, name = "adagrad", ..., loss_scale_factor = NULL)
{
    args <- capture_args(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$Adagrad, args)
}


#' Optimizer that implements the Adam algorithm.
#'
#' @description
#' Adam optimization is a stochastic gradient descent method that is based on
#' adaptive estimation of first-order and second-order moments.
#'
#' According to
#' [Kingma et al., 2014](https://arxiv.org/abs/1412.6980),
#' the method is "*computationally
#' efficient, has little memory requirement, invariant to diagonal rescaling of
#' gradients, and is well suited for problems that are large in terms of
#' data/parameters*".
#'
#' @param learning_rate
#' A float, a
#' [`LearningRateSchedule()`] instance, or
#' a callable that takes no arguments and returns the actual value to
#' use. The learning rate. Defaults to `0.001`.
#'
#' @param beta_1
#' A float value or a constant float tensor, or a callable
#' that takes no arguments and returns the actual value to use. The
#' exponential decay rate for the 1st moment estimates. Defaults to
#' `0.9`.
#'
#' @param beta_2
#' A float value or a constant float tensor, or a callable
#' that takes no arguments and returns the actual value to use. The
#' exponential decay rate for the 2nd moment estimates. Defaults to
#' `0.999`.
#'
#' @param epsilon
#' A small constant for numerical stability. This epsilon is
#' "epsilon hat" in the Kingma and Ba paper (in the formula just before
#' Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults
#' to `1e-7`.
#'
#' @param amsgrad
#' Boolean. Whether to apply AMSGrad variant of this algorithm
#' from the paper "On the Convergence of Adam and beyond". Defaults
#' to `FALSE`.
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
#' Boolean, defaults to `FALSE`. If `TRUE`, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum
#' Float, defaults to 0.99. Only used if `use_ema=TRUE`.
#' This is the momentum to use when computing
#' the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency
#' Int or `NULL`, defaults to `NULL`. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations,
#' we overwrite the model variable by its moving average.
#' If `NULL`, the optimizer
#' does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()`
#' (which updates the model
#' variables in-place). When using the built-in `fit()` training loop,
#' this happens automatically after the last epoch,
#' and you don't need to do anything.
#'
#' @param loss_scale_factor
#' Float or `NULL`. If a float, the scale factor will
#' be multiplied the loss before computing gradients, and the inverse of
#' the scale factor will be multiplied by the gradients before updating
#' variables. Useful for preventing underflow during mixed precision
#' training. Alternately, [`optimizer_loss_scale()`] will
#' automatically set a loss scale factor.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family optimizers
#' @seealso
#' + <https://keras.io/api/optimizers/adam#adam-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam>
#' @tether keras.optimizers.Adam
optimizer_adam <-
function (learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999,
    epsilon = 1e-07, amsgrad = FALSE, weight_decay = NULL, clipnorm = NULL,
    clipvalue = NULL, global_clipnorm = NULL, use_ema = FALSE,
    ema_momentum = 0.99, ema_overwrite_frequency = NULL, name = "adam",
    ..., loss_scale_factor = NULL)
{
    args <- capture_args(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$Adam, args)
}


#' Optimizer that implements the Adamax algorithm.
#'
#' @description
#' Adamax, a variant of Adam based on the infinity norm, is a first-order
#' gradient-based optimization method. Due to its capability of adjusting the
#' learning rate based on data characteristics, it is suited to learn
#' time-variant process, e.g., speech data with dynamically changed noise
#' conditions. Default parameters follow those provided in the paper (see
#' references below).
#'
#' Initialization:
#'
#' ```{r}
#' m <- 0  # Initialize initial 1st moment vector
#' u <- 0  # Initialize the exponentially weighted infinity norm
#' t <- 0  # Initialize timestep
#' ```
#'
#' The update rule for parameter `w` with gradient `g` is described at the end
#' of section 7.1 of the paper (see the referenece section):
#'
#' ```{r, eval=FALSE}
#' t <-  t + 1
#' m <- beta1 * m + (1 - beta) * g
#' u <- max(beta2 * u, abs(g))
#' current_lr <- learning_rate / (1 - beta1 ** t)
#' w <- w - current_lr * m / (u + epsilon)
#' ```
#'
#' # Reference
#' - [Kingma et al., 2014](https://arxiv.org/abs/1412.6980)
#'
#' @param learning_rate
#' A float, a
#' [`LearningRateSchedule()`] instance, or
#' a callable that takes no arguments and returns the actual value to
#' use. The learning rate. Defaults to `0.001`.
#'
#' @param beta_1
#' A float value or a constant float tensor. The exponential decay
#' rate for the 1st moment estimates.
#'
#' @param beta_2
#' A float value or a constant float tensor. The exponential decay
#' rate for the exponentially weighted infinity norm.
#'
#' @param epsilon
#' A small constant for numerical stability.
#'   name: String. The name to use
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
#' Boolean, defaults to FALSE. If TRUE, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum
#' Float, defaults to 0.99. Only used if `use_ema=TRUE`.
#' This is the momentum to use when computing
#' the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency
#' Int or NULL, defaults to NULL. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations,
#' we overwrite the model variable by its moving average.
#' If NULL, the optimizer
#' does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()`
#' (which updates the model
#' variables in-place). When using the built-in `fit()` training loop,
#' this happens automatically after the last epoch,
#' and you don't need to do anything.
#'
#' @param loss_scale_factor
#' Float or `NULL`. If a float, the scale factor will
#' be multiplied the loss before computing gradients, and the inverse of
#' the scale factor will be multiplied by the gradients before updating
#' variables. Useful for preventing underflow during mixed precision
#' training. Alternately, [`optimizer_loss_scale()`] will
#' automatically set a loss scale factor.
#'
#' @param name
#' String, name for the object
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family optimizers
#' @seealso
#' + <https://keras.io/api/optimizers/adamax#adamax-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adamax>
#'
#' @tether keras.optimizers.Adamax
optimizer_adamax <-
function (learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999,
    epsilon = 1e-07, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL,
    global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99,
    ema_overwrite_frequency = NULL, name = "adamax", ..., loss_scale_factor = NULL)
{
    args <- capture_args(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$Adamax, args)
}


#' Optimizer that implements the AdamW algorithm.
#'
#' @description
#' AdamW optimization is a stochastic gradient descent method that is based on
#' adaptive estimation of first-order and second-order moments with an added
#' method to decay weights per the techniques discussed in the paper,
#' 'Decoupled Weight Decay Regularization' by
#' [Loshchilov, Hutter et al., 2019](https://arxiv.org/abs/1711.05101).
#'
#' According to
#' [Kingma et al., 2014](https://arxiv.org/abs/1412.6980),
#' the underying Adam method is "*computationally
#' efficient, has little memory requirement, invariant to diagonal rescaling of
#' gradients, and is well suited for problems that are large in terms of
#' data/parameters*".
#'
#' # References
#' - [Loshchilov et al., 2019](https://arxiv.org/abs/1711.05101)
#' - [Kingma et al., 2014](https://arxiv.org/abs/1412.6980) for `adam`
#' - [Reddi et al., 2018](
#'     https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.
#'
#' @param learning_rate
#' A float, a
#' [`LearningRateSchedule()`] instance, or
#' a callable that takes no arguments and returns the actual value to
#' use. The learning rate. Defaults to `0.001`.
#'
#' @param beta_1
#' A float value or a constant float tensor, or a callable
#' that takes no arguments and returns the actual value to use. The
#' exponential decay rate for the 1st moment estimates.
#' Defaults to `0.9`.
#'
#' @param beta_2
#' A float value or a constant float tensor, or a callable
#' that takes no arguments and returns the actual value to use. The
#' exponential decay rate for the 2nd moment estimates.
#' Defaults to `0.999`.
#'
#' @param epsilon
#' A small constant for numerical stability. This epsilon is
#' "epsilon hat" in the Kingma and Ba paper (in the formula just
#' before Section 2.1), not the epsilon in Algorithm 1 of the paper.
#' Defaults to 1e-7.
#'
#' @param amsgrad
#' Boolean. Whether to apply AMSGrad variant of this algorithm
#' from the paper "On the Convergence of Adam and beyond".
#' Defaults to `FALSE`.
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
#' Boolean, defaults to `FALSE`. If `TRUE`, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum
#' Float, defaults to 0.99. Only used if `use_ema=TRUE`.
#' This is the momentum to use when computing
#' the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) * current_variable_value`.
#'
#' @param ema_overwrite_frequency
#' Int or `NULL`, defaults to `NULL`. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations,
#' we overwrite the model variable by its moving average.
#' If `NULL`, the optimizer
#' does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()`
#' (which updates the model
#' variables in-place). When using the built-in `fit()` training loop,
#' this happens automatically after the last epoch,
#' and you don't need to do anything.
#'
#' @param loss_scale_factor
#' Float or `NULL`. If a float, the scale factor will
#' be multiplied the loss before computing gradients, and the inverse of
#' the scale factor will be multiplied by the gradients before updating
#' variables. Useful for preventing underflow during mixed precision
#' training. Alternately, [`optimizer_loss_scale()`] will
#' automatically set a loss scale factor.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family optimizers
#' @seealso
#' + <https://keras.io/api/optimizers/adamw#adamw-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/AdamW>
#' @tether keras.optimizers.AdamW
optimizer_adam_w <-
function (learning_rate = 0.001, weight_decay = 0.004, beta_1 = 0.9,
    beta_2 = 0.999, epsilon = 1e-07, amsgrad = FALSE, clipnorm = NULL,
    clipvalue = NULL, global_clipnorm = NULL, use_ema = FALSE,
    ema_momentum = 0.99, ema_overwrite_frequency = NULL, name = "adamw",
    ..., loss_scale_factor = NULL)
{
    args <- capture_args(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$AdamW, args)
}


#' Optimizer that implements the FTRL algorithm.
#'
#' @description
#' "Follow The Regularized Leader" (FTRL) is an optimization algorithm
#' developed at Google for click-through rate prediction in the early 2010s. It
#' is most suitable for shallow models with large and sparse feature spaces.
#' The algorithm is described by
#' [McMahan et al., 2013](https://research.google.com/pubs/archive/41159.pdf).
#' The Keras version has support for both online L2 regularization
#' (the L2 regularization described in the paper
#' above) and shrinkage-type L2 regularization
#' (which is the addition of an L2 penalty to the loss function).
#'
#' Initialization:
#'
#' ```{r}
#' n <- 0
#' sigma <- 0
#' z <- 0
#' ```
#'
#' Update rule for one variable `w`:
#'
#' ```{r, eval=FALSE}
#' prev_n <- n
#' n <- n + g^2
#' sigma <- (n^(-lr_power) - prev_n^(-lr_power)) / lr
#' z <- z + g - sigma * w
#' if (abs(z) < lambda_1) {
#'   w <- 0
#' } else {
#'   w <- (sgn(z) * lambda_1 - z) / ((beta + sqrt(n)) / alpha + lambda_2)
#' }
#' ```
#'
#' Notation:
#'
#' - `lr` is the learning rate
#' - `g` is the gradient for the variable
#' - `lambda_1` is the L1 regularization strength
#' - `lambda_2` is the L2 regularization strength
#' - `lr_power` is the power to scale n.
#'
#' Check the documentation for the `l2_shrinkage_regularization_strength`
#' parameter for more details when shrinkage is enabled, in which case gradient
#' is replaced with a gradient with shrinkage.
#'
#' @param learning_rate
#' A float, a
#' [`LearningRateSchedule()`] instance, or
#' a callable that takes no arguments and returns the actual value to
#' use. The learning rate. Defaults to `0.001`.
#'
#' @param learning_rate_power
#' A float value, must be less or equal to zero.
#' Controls how the learning rate decreases during training. Use zero
#' for a fixed learning rate.
#'
#' @param initial_accumulator_value
#' The starting value for accumulators. Only
#' zero or positive values are allowed.
#'
#' @param l1_regularization_strength
#' A float value, must be greater than or equal
#' to zero. Defaults to `0.0`.
#'
#' @param l2_regularization_strength
#' A float value, must be greater than or equal
#' to zero. Defaults to `0.0`.
#'
#' @param l2_shrinkage_regularization_strength
#' A float value, must be greater
#' than or equal to zero. This differs from L2 above in that the L2
#' above is a stabilization penalty, whereas this L2 shrinkage is a
#' magnitude penalty. When input is sparse shrinkage will only happen
#' on the active weights.
#'
#' @param beta
#' A float value, representing the beta value from the paper.
#' Defaults to `0.0`.
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
#' Boolean, defaults to FALSE. If TRUE, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum
#' Float, defaults to 0.99. Only used if `use_ema=TRUE`.
#' This is the momentum to use when computing
#' the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency
#' Int or NULL, defaults to NULL. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations,
#' we overwrite the model variable by its moving average.
#' If NULL, the optimizer
#' does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer$finalize_variable_values()`
#' (which updates the model
#' variables in-place). When using the built-in `fit()` training loop,
#' this happens automatically after the last epoch,
#' and you don't need to do anything.
#'
#' @param loss_scale_factor
#' Float or `NULL`. If a float, the scale factor will
#' be multiplied the loss before computing gradients, and the inverse of
#' the scale factor will be multiplied by the gradients before updating
#' variables. Useful for preventing underflow during mixed precision
#' training. Alternately, `optimizer_loss_scale` will
#' automatically set a loss scale factor.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family optimizers
#' @seealso
#' + <https://keras.io/api/optimizers/ftrl#ftrl-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Ftrl>
#'
#' @tether keras.optimizers.Ftrl
optimizer_ftrl <-
function (learning_rate = 0.001, learning_rate_power = -0.5,
    initial_accumulator_value = 0.1, l1_regularization_strength = 0,
    l2_regularization_strength = 0, l2_shrinkage_regularization_strength = 0,
    beta = 0, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL,
    global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99,
    ema_overwrite_frequency = NULL, name = "ftrl", ..., loss_scale_factor = NULL)
{
    args <- capture_args(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$Ftrl, args)
}


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
#' - [Chen et al., 2023](https://arxiv.org/abs/2302.06675)
#' - [Authors' implementation](
#'     https://github.com/google/automl/tree/master/lion)
#'
#' @param learning_rate
#' A float, a
#' [`LearningRateSchedule()`] instance, or
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
#' Boolean, defaults to `FALSE`. If `TRUE`, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum
#' Float, defaults to 0.99. Only used if `use_ema=TRUE`.
#' This is the momentum to use when computing
#' the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency
#' Int or `NULL`, defaults to `NULL`. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations,
#' we overwrite the model variable by its moving average.
#' If `NULL`, the optimizer
#' does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()`
#' (which updates the model
#' variables in-place). When using the built-in `fit()` training loop,
#' this happens automatically after the last epoch,
#' and you don't need to do anything.
#'
#' @param loss_scale_factor
#' Float or `NULL`. If a float, the scale factor will
#' be multiplied the loss before computing gradients, and the inverse of
#' the scale factor will be multiplied by the gradients before updating
#' variables. Useful for preventing underflow during mixed precision
#' training. Alternately, [`optimizer_loss_scale()`] will
#' automatically set a loss scale factor.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family optimizers
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Lion>
#' @tether keras.optimizers.Lion
optimizer_lion <-
function (learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.99,
    weight_decay = NULL, clipnorm = NULL, clipvalue = NULL, global_clipnorm = NULL,
    use_ema = FALSE, ema_momentum = 0.99, ema_overwrite_frequency = NULL,
    name = "lion", ..., loss_scale_factor = NULL)
{
    args <- capture_args(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$Lion, args)
}


#' An optimizer that dynamically scales the loss to prevent underflow.
#'
#' @description
#' Loss scaling is a technique to prevent numeric underflow in intermediate
#' gradients when float16 is used. To prevent underflow, the loss is multiplied
#' (or "scaled") by a certain factor called the "loss scale", which causes
#' intermediate gradients to be scaled by the loss scale as well. The final
#' gradients are divided (or "unscaled") by the loss scale to bring them back
#' to their original value.
#'
#' `LossScaleOptimizer` wraps another optimizer and applies dynamic loss
#' scaling to it. This loss scale is dynamically updated over time as follows:
#' - On any train step, if a nonfinite gradient is encountered, the loss scale
#'   is halved, and the train step is skipped.
#' - If `dynamic_growth_steps` have ocurred since the last time the loss scale
#'   was updated, and no nonfinite gradients have occurred, the loss scale
#'   is doubled.
#'
#' @param inner_optimizer
#' The keras `Optimizer` instance to wrap.
#'
#' @param initial_scale
#' Float. The initial loss scale. This scale will be updated
#' during training. It is recommended for this to be a very high
#' number, because a loss scale that is too high gets lowered far more
#' quickly than a loss scale that is too low gets raised.
#'
#' @param dynamic_growth_steps
#' Int. How often to update the scale upwards. After
#' every `dynamic_growth_steps` steps with finite gradients, the
#' loss scale is doubled.
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
#' Boolean, defaults to `FALSE`. If `TRUE`, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum
#' Float, defaults to 0.99. Only used if `use_ema=TRUE`.
#' This is the momentum to use when computing
#' the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency
#' Int or `NULL`, defaults to `NULL`. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations,
#' we overwrite the model variable by its moving average.
#' If `NULL`, the optimizer
#' does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer$finalize_variable_values()`
#' (which updates the model
#' variables in-place). When using the built-in `fit()` training loop,
#' this happens automatically after the last epoch,
#' and you don't need to do anything.
#'
#' @param loss_scale_factor
#' Float or `NULL`. If a float, the scale factor will
#' be multiplied the loss before computing gradients, and the inverse of
#' the scale factor will be multiplied by the gradients before updating
#' variables. Useful for preventing underflow during mixed precision
#' training. Alternately, [`optimizer_loss_scale()`] will
#' automatically set a loss scale factor.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family optimizers
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/LossScaleOptimizer>
#' @tether keras.optimizers.LossScaleOptimizer
optimizer_loss_scale <-
function (inner_optimizer, initial_scale = 32768, dynamic_growth_steps = 2000L,
    ..., name = NULL, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL,
    global_clipnorm = NULL, use_ema = NULL, ema_momentum = NULL,
    ema_overwrite_frequency = NULL, loss_scale_factor = NULL)
{
    args <- capture_args(list(dynamic_growth_steps = as_integer,
        ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$LossScaleOptimizer, args)
}


#' Optimizer that implements the Nadam algorithm.
#'
#' @description
#' Much like Adam is essentially RMSprop with momentum, Nadam is Adam with
#' Nesterov momentum.
#'
#' # Reference
#' - [Dozat, 2015](https://cs229.stanford.edu/proj2015/054_report.pdf).
#'
#' @param learning_rate
#' A float, a
#' [`LearningRateSchedule()`] instance, or
#' a callable that takes no arguments and returns the actual value to
#' use. The learning rate. Defaults to `0.001`.
#'
#' @param beta_1
#' A float value or a constant float tensor, or a callable
#' that takes no arguments and returns the actual value to use. The
#' exponential decay rate for the 1st moment estimates.
#' Defaults to `0.9`.
#'
#' @param beta_2
#' A float value or a constant float tensor, or a callable
#' that takes no arguments and returns the actual value to use. The
#' exponential decay rate for the 2nd moment estimates. Defaults to
#' `0.999`.
#'
#' @param epsilon
#' A small constant for numerical stability. This epsilon is
#' "epsilon hat" in the Kingma and Ba paper (in the formula just before
#' Section 2.1), not the epsilon in Algorithm 1 of the paper.
#' Defaults to `1e-7`.
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
#' Boolean, defaults to `FALSE`. If `TRUE`, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum
#' Float, defaults to 0.99. Only used if `use_ema=TRUE`.
#' This is the momentum to use when computing
#' the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency
#' Int or `NULL`, defaults to `NULL`. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations,
#' we overwrite the model variable by its moving average.
#' If `NULL`, the optimizer
#' does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()`
#' (which updates the model
#' variables in-place). When using the built-in `fit()` training loop,
#' this happens automatically after the last epoch,
#' and you don't need to do anything.
#'
#' @param loss_scale_factor
#' Float or `NULL`. If a float, the scale factor will
#' be multiplied the loss before computing gradients, and the inverse of
#' the scale factor will be multiplied by the gradients before updating
#' variables. Useful for preventing underflow during mixed precision
#' training. Alternately, [`optimizer_loss_scale()`] will
#' automatically set a loss scale factor.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family optimizers
#' @seealso
#' + <https://keras.io/api/optimizers/Nadam#nadam-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Nadam>
#' @tether keras.optimizers.Nadam
optimizer_nadam <-
function (learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999,
    epsilon = 1e-07, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL,
    global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99,
    ema_overwrite_frequency = NULL, name = "nadam", ..., loss_scale_factor = NULL)
{
    args <- capture_args(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$Nadam, args)
}


#' Optimizer that implements the RMSprop algorithm.
#'
#' @description
#' The gist of RMSprop is to:
#'
#' - Maintain a moving (discounted) average of the square of gradients
#' - Divide the gradient by the root of this average
#'
#' This implementation of RMSprop uses plain momentum, not Nesterov momentum.
#'
#' The centered version additionally maintains a moving average of the
#' gradients, and uses that average to estimate the variance.
#'
#' # Usage
#' ```{r}
#' opt <- optimizer_rmsprop(learning_rate=0.1)
#' ```
#'
#' # Reference
#' - [Hinton, 2012](
#'     https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
#'
#' @param learning_rate
#' A float, a
#' `learning_rate_schedule_*` instance, or
#' a callable that takes no arguments and returns the actual value to
#' use. The learning rate. Defaults to `0.001`.
#'
#' @param rho
#' float, defaults to 0.9. Discounting factor for the old gradients.
#'
#' @param momentum
#' float, defaults to 0.0. If not 0.0., the optimizer tracks the
#' momentum value, with a decay rate equals to `1 - momentum`.
#'
#' @param epsilon
#' A small constant for numerical stability. This epsilon is
#' "epsilon hat" in the Kingma and Ba paper (in the formula just before
#' Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults
#' to 1e-7.
#'
#' @param centered
#' Boolean. If `TRUE`, gradients are normalized by the estimated
#' variance of the gradient; if FALSE, by the uncentered second moment.
#' Setting this to `TRUE` may help with training, but is slightly more
#' expensive in terms of computation and memory. Defaults to `FALSE`.
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
#' Boolean, defaults to FALSE. If TRUE, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum
#' Float, defaults to 0.99. Only used if `use_ema=TRUE`.
#' This is the momentum to use when computing
#' the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency
#' Int or NULL, defaults to NULL. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations,
#' we overwrite the model variable by its moving average.
#' If NULL, the optimizer
#' does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()`
#' (which updates the model
#' variables in-place). When using the built-in `fit()` training loop,
#' this happens automatically after the last epoch,
#' and you don't need to do anything.
#'
#' @param loss_scale_factor
#' Float or `NULL`. If a float, the scale factor will
#' be multiplied the loss before computing gradients, and the inverse of
#' the scale factor will be multiplied by the gradients before updating
#' variables. Useful for preventing underflow during mixed precision
#' training. Alternately, [`optimizer_loss_scale()`] will
#' automatically set a loss scale factor.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family optimizers
#' @seealso
#' + <https://keras.io/api/optimizers/rmsprop#rmsprop-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop>
#'
#' @tether keras.optimizers.RMSprop
optimizer_rmsprop <-
function (learning_rate = 0.001, rho = 0.9, momentum = 0, epsilon = 1e-07,
    centered = FALSE, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL,
    global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99,
    ema_overwrite_frequency = 100L, name = "rmsprop", ..., loss_scale_factor = NULL)
{
    args <- capture_args(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$RMSprop, args)
}


#' Gradient descent (with momentum) optimizer.
#'
#' @description
#' Update rule for parameter `w` with gradient `g` when `momentum` is 0:
#'
#' ```{r, eval = FALSE}
#' w <- w - learning_rate * g
#' ```
#'
#' Update rule when `momentum` is larger than 0:
#'
#' ```{r, eval = FALSE}
#' velocity <- momentum * velocity - learning_rate * g
#' w <- w + velocity
#' ```
#'
#' When `nesterov=TRUE`, this rule becomes:
#'
#' ```{r, eval = FALSE}
#' velocity <- momentum * velocity - learning_rate * g
#' w <- w + momentum * velocity - learning_rate * g
#' ```
#'
#' @param learning_rate
#' A float, a
#' `learning_rate_schedule_*` instance, or
#' a callable that takes no arguments and returns the actual value to
#' use. The learning rate. Defaults to `0.01`.
#'
#' @param momentum
#' float hyperparameter >= 0 that accelerates gradient descent in
#' the relevant direction and dampens oscillations. 0 is vanilla
#' gradient descent. Defaults to `0.0`.
#'
#' @param nesterov
#' boolean. Whether to apply Nesterov momentum.
#' Defaults to `FALSE`.
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
#' Boolean, defaults to FALSE. If TRUE, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum
#' Float, defaults to 0.99. Only used if `use_ema=TRUE`.
#' This is the momentum to use when computing
#' the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency
#' Int or NULL, defaults to NULL. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations,
#' we overwrite the model variable by its moving average.
#' If NULL, the optimizer
#' does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer$finalize_variable_values()`
#' (which updates the model
#' variables in-place). When using the built-in `fit()` training loop,
#' this happens automatically after the last epoch,
#' and you don't need to do anything.
#'
#' @param loss_scale_factor
#' Float or `NULL`. If a float, the scale factor will
#' be multiplied the loss before computing gradients, and the inverse of
#' the scale factor will be multiplied by the gradients before updating
#' variables. Useful for preventing underflow during mixed precision
#' training. Alternately, `optimizer_loss_scale()` will
#' automatically set a loss scale factor.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family optimizers
#' @seealso
#' + <https://keras.io/api/optimizers/sgd#sgd-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD>
#'
#' @tether keras.optimizers.SGD
optimizer_sgd <-
function (learning_rate = 0.01, momentum = 0, nesterov = FALSE,
    weight_decay = NULL, clipnorm = NULL, clipvalue = NULL, global_clipnorm = NULL,
    use_ema = FALSE, ema_momentum = 0.99, ema_overwrite_frequency = NULL,
    name = "SGD", ..., loss_scale_factor = NULL)
{
    args <- capture_args(list(ema_overwrite_frequency = as_integer))
    do.call(keras$optimizers$SGD, args)
}
