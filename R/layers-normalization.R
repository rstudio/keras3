
#' Batch normalization layer (Ioffe and Szegedy, 2014).
#' 
#' Normalize the activations of the previous layer at each batch, i.e. applies a
#' transformation that maintains the mean activation close to 0 and the
#' activation standard deviation close to 1.
#' 
#' @inheritParams layer_dense
#' 
#' @param axis Integer, the axis that should be normalized (typically the
#'   features axis). For instance, after a `Conv2D` layer with
#'   `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
#' @param momentum Momentum for the moving average.
#' @param epsilon Small float added to variance to avoid dividing by zero.
#' @param center If TRUE, add offset of `beta` to normalized tensor. If FALSE,
#'   `beta` is ignored.
#' @param scale If TRUE, multiply by `gamma`. If FALSE, `gamma` is not used.
#'   When the next layer is linear (also e.g. `nn.relu`), this can be disabled
#'   since the scaling will be done by the next layer.
#' @param beta_initializer Initializer for the beta weight.
#' @param gamma_initializer Initializer for the gamma weight.
#' @param moving_mean_initializer Initializer for the moving mean.
#' @param moving_variance_initializer Initializer for the moving variance.
#' @param beta_regularizer Optional regularizer for the beta weight.
#' @param gamma_regularizer Optional regularizer for the gamma weight.
#' @param beta_constraint Optional constraint for the beta weight.
#' @param gamma_constraint Optional constraint for the gamma weight.
#'   
#' @section Input shape: Arbitrary. Use the keyword argument `input_shape` (list
#'   of integers, does not include the samples axis) when using this layer as
#'   the first layer in a model.
#'   
#' @section Output shape: Same shape as input.
#'   
#' @section References: 
#' - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
#'   
#' @export
layer_batch_normalization <- function(object, axis = -1L, momentum = 0.99, epsilon = 0.001, center = TRUE, scale = TRUE, 
                                      beta_initializer = "zeros", gamma_initializer = "ones", 
                                      moving_mean_initializer = "zeros", moving_variance_initializer = "ones", 
                                      beta_regularizer = NULL, gamma_regularizer = NULL, 
                                      beta_constraint = NULL, gamma_constraint = NULL, 
                                      input_shape = NULL,  batch_input_shape = NULL, batch_size = NULL, 
                                      dtype = NULL, name = NULL, trainable = NULL, weights = NULL) {
  create_layer(keras$layers$BatchNormalization, object, list(
    axis = as.integer(axis),
    momentum = momentum,
    epsilon = epsilon,
    center = center,
    scale = scale,
    beta_initializer = beta_initializer,
    gamma_initializer = gamma_initializer,
    moving_mean_initializer = moving_mean_initializer,
    moving_variance_initializer = moving_variance_initializer,
    beta_regularizer = beta_regularizer,
    gamma_regularizer = gamma_regularizer,
    beta_constraint = beta_constraint,
    gamma_constraint = gamma_constraint,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    weights = weights
  ))
}