
#' Layer that normalizes its inputs
#'
#' @details
#' Batch normalization applies a transformation that maintains the mean output
#' close to 0 and the output standard deviation close to 1.
#'
#' Importantly, batch normalization works differently during training and
#' during inference.
#'
#' **During training** (i.e. when using `fit()` or when calling the layer/model
#' with the argument `training=TRUE`), the layer normalizes its output using
#' the mean and standard deviation of the current batch of inputs. That is to
#' say, for each channel being normalized, the layer returns
#' `gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta`, where:
#'
#' - `epsilon` is small constant (configurable as part of the constructor
#' arguments)
#' - `gamma` is a learned scaling factor (initialized as 1), which
#' can be disabled by passing `scale=FALSE` to the constructor.
#' - `beta` is a learned offset factor (initialized as 0), which
#' can be disabled by passing `center=FALSE` to the constructor.
#'
#' **During inference** (i.e. when using `evaluate()` or `predict()` or when
#' calling the layer/model with the argument `training=FALSE` (which is the
#' default), the layer normalizes its output using a moving average of the
#' mean and standard deviation of the batches it has seen during training. That
#' is to say, it returns
#' `gamma * (batch - self.moving_mean) / sqrt(self.moving_var+epsilon) + beta`.
#'
#' `self$moving_mean` and `self$moving_var` are non-trainable variables that
#' are updated each time the layer in called in training mode, as such:
#'
#' - `moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)`
#' - `moving_var = moving_var * momentum + var(batch) * (1 - momentum)`
#'
#' As such, the layer will only normalize its inputs during inference
#' *after having been trained on data that has similar statistics as the
#' inference data*.
#'
#' When `synchronized=TRUE` is set and if this layer is used within a
#' `tf$distribute` strategy, there will be an `allreduce` call
#' to aggregate batch statistics across all replicas at every
#' training step. Setting `synchronized` has no impact when the model is
#' trained without specifying any distribution strategy.
#'
#' Example usage:
#'
#' ```R
#' strategy <- tf$distribute$MirroredStrategy()
#'
#' with(strategy$scope(), {
#'   model <- keras_model_sequential()
#'   model %>%
#'     layer_dense(16) %>%
#'     layer_batch_normalization(synchronized=TRUE)
#' })
#' ```
#'
#' @param object Layer or model object
#'
#' @param axis Integer, the axis that should be normalized (typically the features
#' axis). For instance, after a `Conv2D` layer with
#' `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
#'
#' @param momentum Momentum for the moving average.
#'
#' @param epsilon Small float added to variance to avoid dividing by zero.
#'
#' @param center If `TRUE`, add offset of `beta` to normalized tensor. If `FALSE`,
#' `beta` is ignored.
#'
#' @param scale If `TRUE`, multiply by `gamma`. If `FALSE`, `gamma` is not used. When
#' the next layer is linear (also e.g. `nn.relu`), this can be disabled
#' since the scaling will be done by the next layer.
#'
#' @param beta_initializer Initializer for the beta weight.
#'
#' @param gamma_initializer Initializer for the gamma weight.
#'
#' @param moving_mean_initializer Initializer for the moving mean.
#'
#' @param moving_variance_initializer Initializer for the moving variance.
#'
#' @param beta_regularizer Optional regularizer for the beta weight.
#'
#' @param gamma_regularizer Optional regularizer for the gamma weight.
#'
#' @param beta_constraint Optional constraint for the beta weight.
#'
#' @param gamma_constraint Optional constraint for the gamma weight.
#'
#' @param synchronized If `TRUE`, synchronizes the global batch statistics (mean and
#' variance) for the layer across all devices at each training step in a
#' distributed training strategy. If `FALSE`, each replica uses its own
#' local batch statistics. Only relevant when used inside a
#' `tf$distribute` strategy.
#' @param ... standard layer arguments.
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization>
#'   +  <https://keras.io/api/layers>
#' @export
layer_batch_normalization <-
  function(object,
           axis = -1L,
           momentum = 0.99,
           epsilon = 0.001,
           center = TRUE,
           scale = TRUE,
           beta_initializer = "zeros",
           gamma_initializer = "ones",
           moving_mean_initializer = "zeros",
           moving_variance_initializer = "ones",
           beta_regularizer = NULL,
           gamma_regularizer = NULL,
           beta_constraint = NULL,
           gamma_constraint = NULL,
           synchronized = FALSE,
           ...) {
    args <- capture_args(match.call(), list(
        axis = as_axis,
        input_shape = normalize_shape,
        batch_input_shape = normalize_shape,
        batch_size = as_nullable_integer,
        virtual_batch_size = as_nullable_integer
      ), ignore = "object"
    )
    create_layer(keras$layers$BatchNormalization, object, args)
  }


#' Layer normalization layer (Ba et al., 2016).
#'
#' Normalize the activations of the previous layer for each given example in a
#' batch independently, rather than across a batch like Batch Normalization. i.e.
#' applies a transformation that maintains the mean activation within each example
#' close to 0 and the activation standard deviation close to 1.
#'
#' Given a tensor inputs, moments are calculated and normalization is performed
#' across the axes specified in axis.
#'
#' @inheritParams layer_dense
#' @param axis Integer or List/Tuple. The axis or axes to normalize across.
#'   Typically this is the features axis/axes. The left-out axes are typically
#'   the batch axis/axes. This argument defaults to -1, the last dimension in
#'   the input.
#' @param epsilon Small float added to variance to avoid dividing by zero.
#'   Defaults to 1e-3
#' @param center If True, add offset of beta to normalized tensor. If False,
#'   beta is ignored. Defaults to True.
#' @param scale If True, multiply by gamma. If False, gamma is not used.
#'   Defaults to True. When the next layer is linear (also e.g. nn.relu), this
#'   can be disabled since the scaling will be done by the next layer.
#' @param beta_initializer Initializer for the beta weight. Defaults to zeros.
#' @param gamma_initializer Initializer for the gamma weight. Defaults to ones.
#' @param beta_regularizer Optional regularizer for the beta weight.
#'   None by default.
#' @param gamma_regularizer Optional regularizer for the gamma weight.
#'   None by default.
#' @param beta_constraint Optional constraint for the beta weight. None by default.
#' @param gamma_constraint Optional constraint for the gamma weight.
#'   None by default.
#' @param trainable Boolean, if True the variables will be marked as trainable.
#'   Defaults to True.
#'
#' @export
layer_layer_normalization <- function(
  object,
  axis=-1,
  epsilon=0.001,
  center=TRUE,
  scale=TRUE,
  beta_initializer="zeros",
  gamma_initializer="ones",
  beta_regularizer=NULL,
  gamma_regularizer=NULL,
  beta_constraint=NULL,
  gamma_constraint=NULL,
  trainable=TRUE,
  name=NULL
) {

  create_layer(keras$layers$LayerNormalization, object, list(
    axis=as.integer(axis),
    epsilon=epsilon,
    center=center,
    scale=scale,
    beta_initializer=beta_initializer,
    gamma_initializer=gamma_initializer,
    beta_regularizer=beta_regularizer,
    gamma_regularizer=gamma_regularizer,
    beta_constraint=beta_constraint,
    gamma_constraint=gamma_constraint,
    trainable=trainable,
    name=name
  ))
}


#' Unit normalization layer
#'
#' @details
#' Normalize a batch of inputs so that each input in the batch has a L2 norm
#' equal to 1 (across the axes specified in `axis`).
#'
#' @inheritParams layer_dense
#' @param axis Integer or list. The axis or axes to normalize across. Typically
#' this is the features axis or axes. The left-out axes are typically the
#' batch axis or axes. Defaults to `-1`, the last dimension in
#' the input.
#' @param ... standard layer arguments.
#'
#' ````r
#' data <- as_tensor(1:6, shape = c(2, 3), dtype = "float32")
#' normalized_data <- data %>% layer_unit_normalization()
#' for(row in 1:2)
#'   normalized_data[row, ] %>%
#'   { sum(.^2) } %>%
#'   print()
#' # tf.Tensor(0.9999999, shape=(), dtype=float32)
#' # tf.Tensor(1.0, shape=(), dtype=float32)
#' ````
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/UnitNormalization>
#'
#' @export
#'
layer_unit_normalization <-
function(object, axis = -1L, ...)
{
  args <- capture_args(match.call(), list(axis = as_axis),
                       ignore = "object")
  create_layer(keras$layers$UnitNormalization, object, args)
}

