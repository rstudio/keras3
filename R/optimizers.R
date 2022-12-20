#' Optimizer that implements the Adadelta algorithm
#'
#' @details
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
#' @param learning_rate Initial value for the learning rate:
#' either a floating point value,
#' or a `tf.keras.optimizers.schedules.LearningRateSchedule` instance.
#' Defaults to 0.001.
#' Note that `Adadelta` tends to benefit from higher initial learning rate
#' values compared to other optimizers.
#' To match the exact form in the original paper, use 1.0.
#'
#' @param rho A `Tensor` or a floating point value. The decay rate. Defaults to
#' 0.95.
#'
#' @param epsilon Small floating point value used to maintain numerical stability.
#' Defaults to 1e-7.
#'
#' @param name String. The name to use
#' for momentum accumulator weights created by
#' the optimizer.
#'
#' @param weight_decay Float, defaults to NULL. If set, weight decay is applied.
#'
#' @param clipnorm Float. If set, the gradient of each weight is individually
#' clipped so that its norm is no higher than this value.
#'
#' @param clipvalue Float. If set, the gradient of each weight is clipped to be no
#' higher than this value.
#'
#' @param global_clipnorm Float. If set, the gradient of all weights is clipped so
#' that their global norm is no higher than this value.
#'
#' @param use_ema Boolean, defaults to FALSE. If TRUE, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum Float, defaults to 0.99. Only used if `use_ema=TRUE`. This is  # noqa: E501
#' the momentum to use when computing the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency Int or NULL, defaults to NULL. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations, we
#' overwrite the model variable by its moving average. If NULL, the optimizer  # noqa: E501
#'  does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()` (which updates the model  # noqa: E501
#' variables in-place). When using the built-in `fit()` training loop, this
#' happens automatically after the last epoch, and you don't need to do
#' anything.
#'
#' @param jit_compile Boolean, defaults to TRUE. If TRUE, the optimizer will use XLA  # noqa: E501
#' compilation. If no GPU device is found, this flag will be ignored.
#' @param ... Used for backward and forward compatibility
#'
#' @family optimizers
#' @return Optimizer for use with \code{\link{compile.keras.engine.training.Model}}.
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/Adadelta>
#' @export
optimizer_adadelta <-
function(learning_rate = 0.001, rho = 0.95, epsilon = 1e-07,
weight_decay = NULL, clipnorm = NULL, clipvalue = NULL, global_clipnorm = NULL,
use_ema = FALSE, ema_momentum = 0.99, ema_overwrite_frequency = NULL,
jit_compile = TRUE, name = "Adadelta", ...)
{
args <- capture_args(match.call(), NULL)
do.call(keras$optimizers$Adadelta, args)
}

#' Optimizer that implements the Adagrad algorithm
#'
#' @details
#' Adagrad is an optimizer with parameter-specific learning rates,
#' which are adapted relative to how frequently a parameter gets
#' updated during training. The more updates a parameter receives,
#' the smaller the updates.
#'
#' @param learning_rate Initial value for the learning rate:
#' either a floating point value,
#' or a `tf.keras.optimizers.schedules.LearningRateSchedule` instance.
#' Defaults to 0.001.
#' Note that `Adagrad` tends to benefit from higher initial learning rate
#' values compared to other optimizers.
#' To match the exact form in the original paper, use 1.0.
#'
#' @param initial_accumulator_value Floating point value.
#' Starting value for the accumulators (per-parameter momentum values).
#' Must be non-negative.
#'
#' @param epsilon Small floating point value used to maintain numerical stability.
#'
#' @param name String. The name to use
#' for momentum accumulator weights created by
#' the optimizer.
#'
#' @param weight_decay Float, defaults to NULL. If set, weight decay is applied.
#'
#' @param clipnorm Float. If set, the gradient of each weight is individually
#' clipped so that its norm is no higher than this value.
#'
#' @param clipvalue Float. If set, the gradient of each weight is clipped to be no
#' higher than this value.
#'
#' @param global_clipnorm Float. If set, the gradient of all weights is clipped so
#' that their global norm is no higher than this value.
#'
#' @param use_ema Boolean, defaults to FALSE. If TRUE, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum Float, defaults to 0.99. Only used if `use_ema=TRUE`. This is  # noqa: E501
#' the momentum to use when computing the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency Int or NULL, defaults to NULL. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations, we
#' overwrite the model variable by its moving average. If NULL, the optimizer  # noqa: E501
#'  does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()` (which updates the model  # noqa: E501
#' variables in-place). When using the built-in `fit()` training loop, this
#' happens automatically after the last epoch, and you don't need to do
#' anything.
#'
#' @param jit_compile Boolean, defaults to TRUE. If TRUE, the optimizer will use XLA  # noqa: E501
#' compilation. If no GPU device is found, this flag will be ignored.
#' @param ... Used for backward and forward compatibility
#'
#' @family optimizers
#' @return Optimizer for use with \code{\link{compile.keras.engine.training.Model}}.
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/Adagrad>
#' @export
optimizer_adagrad <-
function(learning_rate = 0.001, initial_accumulator_value = 0.1,
epsilon = 1e-07, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL,
global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99,
ema_overwrite_frequency = NULL, jit_compile = TRUE, name = "Adagrad",
...)
{
args <- capture_args(match.call(), NULL)
do.call(keras$optimizers$Adagrad, args)
}

#' Optimizer that implements the Adam algorithm
#'
#' @details
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
#' @param learning_rate A `tf.Tensor`, floating point value, a schedule that is a
#' `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
#' that takes no arguments and returns the actual value to use. The
#' learning rate. Defaults to 0.001.
#'
#' @param beta_1 A float value or a constant float tensor, or a callable
#' that takes no arguments and returns the actual value to use. The
#' exponential decay rate for the 1st moment estimates. Defaults to 0.9.
#'
#' @param beta_2 A float value or a constant float tensor, or a callable
#' that takes no arguments and returns the actual value to use. The
#' exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
#'
#' @param epsilon A small constant for numerical stability. This epsilon is
#' "epsilon hat" in the Kingma and Ba paper (in the formula just before
#' Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
#' 1e-7.
#'
#' @param amsgrad Boolean. Whether to apply AMSGrad variant of this algorithm from
#' the paper "On the Convergence of Adam and beyond". Defaults to `FALSE`.
#'
#' @param name String. The name to use
#' for momentum accumulator weights created by
#' the optimizer.
#'
#' @param weight_decay Float, defaults to NULL. If set, weight decay is applied.
#'
#' @param clipnorm Float. If set, the gradient of each weight is individually
#' clipped so that its norm is no higher than this value.
#'
#' @param clipvalue Float. If set, the gradient of each weight is clipped to be no
#' higher than this value.
#'
#' @param global_clipnorm Float. If set, the gradient of all weights is clipped so
#' that their global norm is no higher than this value.
#'
#' @param use_ema Boolean, defaults to FALSE. If TRUE, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum Float, defaults to 0.99. Only used if `use_ema=TRUE`. This is  # noqa: E501
#' the momentum to use when computing the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency Int or NULL, defaults to NULL. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations, we
#' overwrite the model variable by its moving average. If NULL, the optimizer  # noqa: E501
#'  does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()` (which updates the model  # noqa: E501
#' variables in-place). When using the built-in `fit()` training loop, this
#' happens automatically after the last epoch, and you don't need to do
#' anything.
#'
#' @param jit_compile Boolean, defaults to TRUE. If TRUE, the optimizer will use XLA  # noqa: E501
#' compilation. If no GPU device is found, this flag will be ignored.
#' @param ... Used for backward and forward compatibility
#'
#' @family optimizers
#' @return Optimizer for use with \code{\link{compile.keras.engine.training.Model}}.
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam>
#' @export
optimizer_adam <-
function(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999,
epsilon = 1e-07, amsgrad = FALSE, weight_decay = NULL, clipnorm = NULL,
clipvalue = NULL, global_clipnorm = NULL, use_ema = FALSE,
ema_momentum = 0.99, ema_overwrite_frequency = NULL, jit_compile = TRUE,
name = "Adam", ...)
{
args <- capture_args(match.call(), NULL)
do.call(keras$optimizers$Adam, args)
}

#' Optimizer that implements the Adamax algorithm
#'
#' @details
#' Adamax, a variant of Adam based on the infinity norm, is a first-order
#' gradient-based optimization method. Due to its capability of adjusting the
#' learning rate based on data characteristics, it is suited to learn
#' time-variant process, e.g., speech data with dynamically changed noise
#' conditions. Default parameters follow those provided in the paper (see
#' references below).
#'
#' Initialization:
#'
#' ```python
#' m = 0  # Initialize initial 1st moment vector
#' u = 0  # Initialize the exponentially weighted infinity norm
#' t = 0  # Initialize timestep
#' ```
#'
#' The update rule for parameter `w` with gradient `g` is described at the end
#' of section 7.1 of the paper (see the referenece section):
#'
#' ```python
#' t += 1
#' m = beta1 * m + (1 - beta) * g
#' u = max(beta2 * u, abs(g))
#' current_lr = learning_rate / (1 - beta1 ** t)
#' w = w - current_lr * m / (u + epsilon)
#' ```
#'
#' @param learning_rate A `tf.Tensor`, floating point value, a schedule that is a
#' `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
#' that takes no arguments and returns the actual value to use. The
#' learning rate. Defaults to 0.001.
#'
#' @param beta_1 A float value or a constant float tensor. The exponential decay
#' rate for the 1st moment estimates.
#'
#' @param beta_2 A float value or a constant float tensor. The exponential decay
#' rate for the exponentially weighted infinity norm.
#'
#' @param epsilon A small constant for numerical stability.
#'
#' @param name String. The name to use
#' for momentum accumulator weights created by
#' the optimizer.
#'
#' @param weight_decay Float, defaults to NULL. If set, weight decay is applied.
#'
#' @param clipnorm Float. If set, the gradient of each weight is individually
#' clipped so that its norm is no higher than this value.
#'
#' @param clipvalue Float. If set, the gradient of each weight is clipped to be no
#' higher than this value.
#'
#' @param global_clipnorm Float. If set, the gradient of all weights is clipped so
#' that their global norm is no higher than this value.
#'
#' @param use_ema Boolean, defaults to FALSE. If TRUE, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum Float, defaults to 0.99. Only used if `use_ema=TRUE`. This is  # noqa: E501
#' the momentum to use when computing the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency Int or NULL, defaults to NULL. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations, we
#' overwrite the model variable by its moving average. If NULL, the optimizer  # noqa: E501
#'  does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()` (which updates the model  # noqa: E501
#' variables in-place). When using the built-in `fit()` training loop, this
#' happens automatically after the last epoch, and you don't need to do
#' anything.
#'
#' @param jit_compile Boolean, defaults to TRUE. If TRUE, the optimizer will use XLA  # noqa: E501
#' compilation. If no GPU device is found, this flag will be ignored.
#' @param ... Used for backward and forward compatibility
#'
#' @family optimizers
#' @return Optimizer for use with \code{\link{compile.keras.engine.training.Model}}.
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/Adamax>
#' @export
optimizer_adamax <-
function(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999,
epsilon = 1e-07, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL,
global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99,
ema_overwrite_frequency = NULL, jit_compile = TRUE, name = "Adamax",
...)
{
args <- capture_args(match.call(), NULL)
do.call(keras$optimizers$Adamax, args)
}

#' Optimizer that implements the FTRL algorithm
#'
#' @details
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
#' ```python
#' n = 0
#' sigma = 0
#' z = 0
#' ```
#'
#' Update rule for one variable `w`:
#'
#' ```python
#' prev_n = n
#' n = n + g ** 2
#' sigma = (n ** -lr_power - prev_n ** -lr_power) / lr
#' z = z + g - sigma * w
#' if abs(z) < lambda_1:
#'   w = 0
#' else:
#'   w = (sgn(z) * lambda_1 - z) / ((beta + sqrt(n)) / alpha + lambda_2)
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
#' @param learning_rate A `Tensor`, floating point value, a schedule that is a
#' `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable that
#' takes no arguments and returns the actual value to use. The learning
#' rate.  Defaults to 0.001.
#'
#' @param learning_rate_power A float value, must be less or equal to zero.
#' Controls how the learning rate decreases during training. Use zero for a
#' fixed learning rate.
#'
#' @param initial_accumulator_value The starting value for accumulators. Only zero
#' or positive values are allowed.
#'
#' @param l1_regularization_strength A float value, must be greater than or equal
#' to zero. Defaults to 0.0.
#'
#' @param l2_regularization_strength A float value, must be greater than or equal
#' to zero. Defaults to 0.0.
#'
#' @param l2_shrinkage_regularization_strength A float value, must be greater than
#' or equal to zero. This differs from L2 above in that the L2 above is a
#' stabilization penalty, whereas this L2 shrinkage is a magnitude penalty.
#' When input is sparse shrinkage will only happen on the active weights.
#'
#' @param beta A float value, representing the beta value from the paper. Defaults
#' to 0.0.
#'
#' @param name String. The name to use
#' for momentum accumulator weights created by
#' the optimizer.
#'
#' @param weight_decay Float, defaults to NULL. If set, weight decay is applied.
#'
#' @param clipnorm Float. If set, the gradient of each weight is individually
#' clipped so that its norm is no higher than this value.
#'
#' @param clipvalue Float. If set, the gradient of each weight is clipped to be no
#' higher than this value.
#'
#' @param global_clipnorm Float. If set, the gradient of all weights is clipped so
#' that their global norm is no higher than this value.
#'
#' @param use_ema Boolean, defaults to FALSE. If TRUE, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum Float, defaults to 0.99. Only used if `use_ema=TRUE`. This is  # noqa: E501
#' the momentum to use when computing the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency Int or NULL, defaults to NULL. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations, we
#' overwrite the model variable by its moving average. If NULL, the optimizer  # noqa: E501
#'  does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()` (which updates the model  # noqa: E501
#' variables in-place). When using the built-in `fit()` training loop, this
#' happens automatically after the last epoch, and you don't need to do
#' anything.
#'
#' @param jit_compile Boolean, defaults to TRUE. If TRUE, the optimizer will use XLA  # noqa: E501
#' compilation. If no GPU device is found, this flag will be ignored.
#' @param ... Used for backward and forward compatibility
#'
#' @family optimizers
#' @return Optimizer for use with \code{\link{compile.keras.engine.training.Model}}.
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/Ftrl>
#' @export
optimizer_ftrl <-
function(learning_rate = 0.001, learning_rate_power = -0.5,
initial_accumulator_value = 0.1, l1_regularization_strength = 0,
l2_regularization_strength = 0, l2_shrinkage_regularization_strength = 0,
beta = 0, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL,
global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99,
ema_overwrite_frequency = NULL, jit_compile = TRUE, name = "Ftrl",
...)
{
args <- capture_args(match.call(), NULL)
do.call(keras$optimizers$Ftrl, args)
}

#' Optimizer that implements the Nadam algorithm
#'
#' @details
#' Much like Adam is essentially RMSprop with momentum, Nadam is Adam with
#' Nesterov momentum.
#'
#' @param learning_rate A `tf.Tensor`, floating point value, a schedule that is a
#' `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
#' that takes no arguments and returns the actual value to use. The
#' learning rate. Defaults to 0.001.
#'
#' @param beta_1 A float value or a constant float tensor, or a callable
#' that takes no arguments and returns the actual value to use. The
#' exponential decay rate for the 1st moment estimates. Defaults to 0.9.
#'
#' @param beta_2 A float value or a constant float tensor, or a callable
#' that takes no arguments and returns the actual value to use. The
#' exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
#'
#' @param epsilon A small constant for numerical stability. This epsilon is
#' "epsilon hat" in the Kingma and Ba paper (in the formula just before
#' Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
#' 1e-7.
#'
#' @param name String. The name to use
#' for momentum accumulator weights created by
#' the optimizer.
#'
#' @param weight_decay Float, defaults to NULL. If set, weight decay is applied.
#'
#' @param clipnorm Float. If set, the gradient of each weight is individually
#' clipped so that its norm is no higher than this value.
#'
#' @param clipvalue Float. If set, the gradient of each weight is clipped to be no
#' higher than this value.
#'
#' @param global_clipnorm Float. If set, the gradient of all weights is clipped so
#' that their global norm is no higher than this value.
#'
#' @param use_ema Boolean, defaults to FALSE. If TRUE, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum Float, defaults to 0.99. Only used if `use_ema=TRUE`. This is  # noqa: E501
#' the momentum to use when computing the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency Int or NULL, defaults to NULL. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations, we
#' overwrite the model variable by its moving average. If NULL, the optimizer  # noqa: E501
#'  does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()` (which updates the model  # noqa: E501
#' variables in-place). When using the built-in `fit()` training loop, this
#' happens automatically after the last epoch, and you don't need to do
#' anything.
#'
#' @param jit_compile Boolean, defaults to TRUE. If TRUE, the optimizer will use XLA  # noqa: E501
#' compilation. If no GPU device is found, this flag will be ignored.
#' @param ... Used for backward and forward compatibility
#'
#' @family optimizers
#' @return Optimizer for use with \code{\link{compile.keras.engine.training.Model}}.
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/Nadam>
#' @export
optimizer_nadam <-
function(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999,
epsilon = 1e-07, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL,
global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99,
ema_overwrite_frequency = NULL, jit_compile = TRUE, name = "Nadam",
...)
{
args <- capture_args(match.call(), NULL)
do.call(keras$optimizers$Nadam, args)
}

#' Optimizer that implements the RMSprop algorithm
#'
#' @details
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
#' @param learning_rate Initial value for the learning rate:
#' either a floating point value,
#' or a `tf.keras.optimizers.schedules.LearningRateSchedule` instance.
#' Defaults to 0.001.
#'
#' @param rho float, defaults to 0.9. Discounting factor for the old gradients.
#'
#' @param momentum float, defaults to 0.0. If not 0.0., the optimizer tracks the
#' momentum value, with a decay rate equals to `1 - momentum`.
#'
#' @param epsilon A small constant for numerical stability. This epsilon is
#' "epsilon hat" in the Kingma and Ba paper (in the formula just before
#' Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
#' 1e-7.
#'
#' @param centered Boolean. If `TRUE`, gradients are normalized by the estimated
#' variance of the gradient; if FALSE, by the uncentered second moment.
#' Setting this to `TRUE` may help with training, but is slightly more
#' expensive in terms of computation and memory. Defaults to `FALSE`.
#'
#' @param name String. The name to use
#' for momentum accumulator weights created by
#' the optimizer.
#'
#' @param weight_decay Float, defaults to NULL. If set, weight decay is applied.
#'
#' @param clipnorm Float. If set, the gradient of each weight is individually
#' clipped so that its norm is no higher than this value.
#'
#' @param clipvalue Float. If set, the gradient of each weight is clipped to be no
#' higher than this value.
#'
#' @param global_clipnorm Float. If set, the gradient of all weights is clipped so
#' that their global norm is no higher than this value.
#'
#' @param use_ema Boolean, defaults to FALSE. If TRUE, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum Float, defaults to 0.99. Only used if `use_ema=TRUE`. This is  # noqa: E501
#' the momentum to use when computing the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency Int or NULL, defaults to NULL. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations, we
#' overwrite the model variable by its moving average. If NULL, the optimizer  # noqa: E501
#'  does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()` (which updates the model  # noqa: E501
#' variables in-place). When using the built-in `fit()` training loop, this
#' happens automatically after the last epoch, and you don't need to do
#' anything.
#'
#' @param jit_compile Boolean, defaults to TRUE. If TRUE, the optimizer will use XLA  # noqa: E501
#' compilation. If no GPU device is found, this flag will be ignored.
#' @param ... Used for backward and forward compatibility
#'
#' @family optimizers
#' @return Optimizer for use with \code{\link{compile.keras.engine.training.Model}}.
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/RMSprop>
#' @export
optimizer_rmsprop <-
function(learning_rate = 0.001, rho = 0.9, momentum = 0, epsilon = 1e-07,
centered = FALSE, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL,
global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99,
ema_overwrite_frequency = 100L, jit_compile = TRUE, name = "RMSprop",
...)
{
args <- capture_args(match.call(), list(ema_overwrite_frequency = as.integer))
do.call(keras$optimizers$RMSprop, args)
}

#' Gradient descent (with momentum) optimizer
#'
#' @details
#' Update rule for parameter `w` with gradient `g` when `momentum` is 0:
#'
#' ```python
#' w = w - learning_rate * g
#' ```
#'
#' Update rule when `momentum` is larger than 0:
#'
#' ```python
#' velocity = momentum * velocity - learning_rate * g
#' w = w + velocity
#' ```
#'
#' When `nesterov=TRUE`, this rule becomes:
#'
#' ```python
#' velocity = momentum * velocity - learning_rate * g
#' w = w + momentum * velocity - learning_rate * g
#' ```
#'
#' @param learning_rate A `Tensor`, floating point value, or a schedule that is a
#' `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
#' that takes no arguments and returns the actual value to use. The
#' learning rate. Defaults to 0.001.
#'
#' @param momentum float hyperparameter >= 0 that accelerates gradient descent in
#' the relevant direction and dampens oscillations. Defaults to 0, i.e.,
#' vanilla gradient descent.
#'
#' @param nesterov boolean. Whether to apply Nesterov momentum.
#' Defaults to `FALSE`.
#'
#' @param name String. The name to use
#' for momentum accumulator weights created by
#' the optimizer.
#'
#' @param weight_decay Float, defaults to NULL. If set, weight decay is applied.
#'
#' @param clipnorm Float. If set, the gradient of each weight is individually
#' clipped so that its norm is no higher than this value.
#'
#' @param clipvalue Float. If set, the gradient of each weight is clipped to be no
#' higher than this value.
#'
#' @param global_clipnorm Float. If set, the gradient of all weights is clipped so
#' that their global norm is no higher than this value.
#'
#' @param use_ema Boolean, defaults to FALSE. If TRUE, exponential moving average
#' (EMA) is applied. EMA consists of computing an exponential moving
#' average of the weights of the model (as the weight values change after
#' each training batch), and periodically overwriting the weights with
#' their moving average.
#'
#' @param ema_momentum Float, defaults to 0.99. Only used if `use_ema=TRUE`. This is  # noqa: E501
#' the momentum to use when computing the EMA of the model's weights:
#' `new_average = ema_momentum * old_average + (1 - ema_momentum) *
#' current_variable_value`.
#'
#' @param ema_overwrite_frequency Int or NULL, defaults to NULL. Only used if
#' `use_ema=TRUE`. Every `ema_overwrite_frequency` steps of iterations, we
#' overwrite the model variable by its moving average. If NULL, the optimizer  # noqa: E501
#'  does not overwrite model variables in the middle of training, and you
#' need to explicitly overwrite the variables at the end of training
#' by calling `optimizer.finalize_variable_values()` (which updates the model  # noqa: E501
#' variables in-place). When using the built-in `fit()` training loop, this
#' happens automatically after the last epoch, and you don't need to do
#' anything.
#'
#' @param amsgrad ignored.
#'
#' @param jit_compile Boolean, defaults to TRUE. If TRUE, the optimizer will use XLA  # noqa: E501
#' compilation. If no GPU device is found, this flag will be ignored.
#' @param ... Used for backward and forward compatibility
#'
#' @family optimizers
#' @return Optimizer for use with \code{\link{compile.keras.engine.training.Model}}.
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/SGD>
#' @export
optimizer_sgd <-
function(learning_rate = 0.01, momentum = 0, nesterov = FALSE,
amsgrad = FALSE, weight_decay = NULL, clipnorm = NULL, clipvalue = NULL,
global_clipnorm = NULL, use_ema = FALSE, ema_momentum = 0.99,
ema_overwrite_frequency = NULL, jit_compile = TRUE, name = "SGD",
...)
{
args <- capture_args(match.call(), NULL)
do.call(keras$optimizers$SGD, args)
}
