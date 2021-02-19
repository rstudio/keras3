
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
#' @param momentum Momentum for the moving mean and the moving variance.
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
#' @param renorm Whether to use Batch Renormalization
#'   (https://arxiv.org/abs/1702.03275). This adds extra variables during
#'   training. The inference is the same for either value of this parameter.
#' @param renorm_clipping A named list or dictionary that may map keys `rmax`,
#'   `rmin`, `dmax` to scalar Tensors used to clip the renorm correction. The
#'   correction `(r, d)` is used as `corrected_value = normalized_value * r + d`,
#'   with `r` clipped to `[rmin, rmax]`, and `d` to `[-dmax, dmax]`. Missing `rmax`,
#'   `rmin`, `dmax` are set to `Inf`, `0`, `Inf`, `respectively`.
#' @param renorm_momentum Momentum used to update the moving means and standard
#'   deviations with renorm. Unlike momentum, this affects training and should
#'   be neither too small (which would add noise) nor too large (which would
#'   give stale estimates). Note that momentum is still applied to get the means
#'   and variances for inference.
#' @param fused `TRUE`, use a faster, fused implementation, or raise a ValueError
#'   if the fused implementation cannot be used. If `NULL`, use the faster
#'   implementation if possible. If `FALSE`, do not use the fused implementation.
#' @param virtual_batch_size An integer. By default, virtual_batch_size is `NULL`,
#'   which means batch normalization is performed across the whole batch.
#'   When virtual_batch_size is not `NULL`, instead perform "Ghost Batch 
#'   Normalization", which creates virtual sub-batches which are each normalized
#'   separately (with shared gamma, beta, and moving statistics). Must divide
#'   the actual `batch size` during execution.
#' @param adjustment A function taking the Tensor containing the (dynamic) shape
#'   of the input tensor and returning a pair `(scale, bias)` to apply to the
#'   normalized values `(before gamma and beta)`, only during training.
#'   For example, if `axis==-1`, 
#'   \code{adjustment <- function(shape) {
#'     tuple(tf$random$uniform(shape[-1:NULL, style = "python"], 0.93, 1.07),
#'           tf$random$uniform(shape[-1:NULL, style = "python"], -0.1, 0.1))
#'    }}
#'   will scale the normalized value
#'   by up to 7% up or down, then shift the result by up to 0.1 (with 
#'   independent scaling and bias for each feature but shared across all examples),
#'   and finally apply gamma and/or beta. If `NULL`, no adjustment is applied.
#'   Cannot be specified if virtual_batch_size is specified.
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
                                      beta_regularizer = NULL, gamma_regularizer = NULL, beta_constraint = NULL, 
                                      gamma_constraint = NULL, renorm = FALSE, renorm_clipping = NULL, 
                                      renorm_momentum = 0.99, fused = NULL, virtual_batch_size = NULL,
                                      adjustment = NULL, input_shape = NULL,  batch_input_shape = NULL, 
                                      batch_size = NULL, dtype = NULL, name = NULL, trainable = NULL, weights = NULL) {
  
  stopifnot(is.null(adjustment) || is.function(adjustment))

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
    renorm = renorm,
    renorm_clipping = renorm_clipping,
    renorm_momentum = renorm_momentum,
    fused = fused,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    dtype = dtype,
    name = name,
    trainable = trainable,
    virtual_batch_size = as_nullable_integer(virtual_batch_size),
    adjustment = adjustment,
    weights = weights
  ))
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
