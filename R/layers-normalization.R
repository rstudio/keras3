


#' Layer that normalizes its inputs.
#'
#' @description
#' Batch normalization applies a transformation that maintains the mean output
#' close to 0 and the output standard deviation close to 1.
#'
#' Importantly, batch normalization works differently during training and
#' during inference.
#'
#' **During training** (i.e. when using `fit()` or when calling the layer/model
#' with the argument `training = TRUE`), the layer normalizes its output using
#' the mean and standard deviation of the current batch of inputs. That is to
#' say, for each channel being normalized, the layer returns
#' `gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta`, where:
#'
#' - `epsilon` is small constant (configurable as part of the constructor
#' arguments)
#' - `gamma` is a learned scaling factor (initialized as 1), which
#' can be disabled by passing `scale = FALSE` to the constructor.
#' - `beta` is a learned offset factor (initialized as 0), which
#' can be disabled by passing `center = FALSE` to the constructor.
#'
#' **During inference** (i.e. when using `evaluate()` or `predict()` or when
#' calling the layer/model with the argument `training = FALSE` (which is the
#' default), the layer normalizes its output using a moving average of the
#' mean and standard deviation of the batches it has seen during training. That
#' is to say, it returns
#' `gamma * (batch - self$moving_mean) / sqrt(self$moving_var+epsilon) + beta`.
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
#' **About setting `layer$trainable <- FALSE` on a `BatchNormalization` layer:**
#'
#' The meaning of setting `layer$trainable <- FALSE` is to freeze the layer,
#' i.e. its internal state will not change during training:
#' its trainable weights will not be updated
#' during `fit()` or `train_on_batch()`, and its state updates will not be run.
#'
#' Usually, this does not necessarily mean that the layer is run in inference
#' mode (which is normally controlled by the `training` argument that can
#' be passed when calling a layer). "Frozen state" and "inference mode"
#' are two separate concepts.
#'
#' However, in the case of the `BatchNormalization` layer, **setting
#' `trainable <- FALSE` on the layer means that the layer will be
#' subsequently run in inference mode** (meaning that it will use
#' the moving mean and the moving variance to normalize the current batch,
#' rather than using the mean and variance of the current batch).
#'
#' Note that:
#'
#' - Setting `trainable` on an model containing other layers will recursively
#'     set the `trainable` value of all inner layers.
#' - If the value of the `trainable` attribute is changed after calling
#'     `compile()` on a model, the new value doesn't take effect for this model
#'     until `compile()` is called again.
#'
#' # Call Arguments
#' - `inputs`: Input tensor (of any rank).
#' - `training`: R boolean indicating whether the layer should behave in
#'     training mode or in inference mode.
#'     - `training = TRUE`: The layer will normalize its inputs using
#'     the mean and variance of the current batch of inputs.
#'     - `training = FALSE`: The layer will normalize its inputs using
#'     the mean and variance of its moving statistics, learned during
#'     training.
#' - `mask`: Binary tensor of shape broadcastable to `inputs` tensor, with
#'     `TRUE` values indicating the positions for which mean and variance
#'     should be computed. Masked elements of the current inputs are not
#'     taken into account for mean and variance computation during
#'     training. Any prior unmasked element values will be taken into
#'     account until their momentum expires.
#'
#' # Reference
#' - [Ioffe and Szegedy, 2015](https://arxiv.org/abs/1502.03167).
#'
#' @param axis
#' Integer, the axis that should be normalized
#' (typically the features axis). For instance, after a `Conv2D` layer
#' with `data_format = "channels_first"`, use `axis = 2`.
#'
#' @param momentum
#' Momentum for the moving average.
#'
#' @param epsilon
#' Small float added to variance to avoid dividing by zero.
#'
#' @param center
#' If `TRUE`, add offset of `beta` to normalized tensor.
#' If `FALSE`, `beta` is ignored.
#'
#' @param scale
#' If `TRUE`, multiply by `gamma`. If `FALSE`, `gamma` is not used.
#' When the next layer is linear this can be disabled
#' since the scaling will be done by the next layer.
#'
#' @param beta_initializer
#' Initializer for the beta weight.
#'
#' @param gamma_initializer
#' Initializer for the gamma weight.
#'
#' @param moving_mean_initializer
#' Initializer for the moving mean.
#'
#' @param moving_variance_initializer
#' Initializer for the moving variance.
#'
#' @param beta_regularizer
#' Optional regularizer for the beta weight.
#'
#' @param gamma_regularizer
#' Optional regularizer for the gamma weight.
#'
#' @param beta_constraint
#' Optional constraint for the beta weight.
#'
#' @param gamma_constraint
#' Optional constraint for the gamma weight.
#'
#' @param synchronized
#' Only applicable with the TensorFlow backend.
#' If `TRUE`, synchronizes the global batch statistics (mean and
#' variance) for the layer across all devices at each training step
#' in a distributed training strategy.
#' If `FALSE`, each replica uses its own local batch statistics.
#'
#' @param ...
#' Base layer keyword arguments (e.g. `name` and `dtype`).
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @inherit layer_dense return
#' @export
#' @family normalization layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/normalization_layers/batch_normalization#batchnormalization-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization>
#' @tether keras.layers.BatchNormalization
layer_batch_normalization <-
function (object, axis = -1L, momentum = 0.99, epsilon = 0.001,
    center = TRUE, scale = TRUE, beta_initializer = "zeros",
    gamma_initializer = "ones", moving_mean_initializer = "zeros",
    moving_variance_initializer = "ones", beta_regularizer = NULL,
    gamma_regularizer = NULL, beta_constraint = NULL, gamma_constraint = NULL,
    synchronized = FALSE, ...)
{
    args <- capture_args(list(axis = as_axis, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$BatchNormalization, object, args)
}


#' Group normalization layer.
#'
#' @description
#' Group Normalization divides the channels into groups and computes
#' within each group the mean and variance for normalization.
#' Empirically, its accuracy is more stable than batch norm in a wide
#' range of small batch sizes, if learning rate is adjusted linearly
#' with batch sizes.
#'
#' Relation to Layer Normalization:
#' If the number of groups is set to 1, then this operation becomes nearly
#' identical to Layer Normalization (see Layer Normalization docs for details).
#'
#' Relation to Instance Normalization:
#' If the number of groups is set to the input dimension (number of groups is
#' equal to number of channels), then this operation becomes identical to
#' Instance Normalization. You can achieve this via `groups=-1`.
#'
#' # Input Shape
#' Arbitrary. Use the keyword argument
#' `input_shape` (tuple of integers, does not include the samples
#' axis) when using this layer as the first layer in a model.
#'
#' # Output Shape
#' Same shape as input.
#' **kwargs: Base layer keyword arguments (e.g. `name` and `dtype`).
#'
#' # Reference
#' - [Yuxin Wu & Kaiming He, 2018](https://arxiv.org/abs/1803.08494)
#'
#' @param groups
#' Integer, the number of groups for Group Normalization. Can be in
#' the range `[1, N]` where N is the input dimension. The input
#' dimension must be divisible by the number of groups.
#' Defaults to 32.
#'
#' @param axis
#' Integer or List/Tuple. The axis or axes to normalize across.
#' Typically, this is the features axis/axes. The left-out axes are
#' typically the batch axis/axes. -1 is the last dimension in the
#' input. Defaults to `-1`.
#'
#' @param epsilon
#' Small float added to variance to avoid dividing by zero.
#' Defaults to 1e-3.
#'
#' @param center
#' If `TRUE`, add offset of `beta` to normalized tensor.
#' If `FALSE`, `beta` is ignored. Defaults to `TRUE`.
#'
#' @param scale
#' If `TRUE`, multiply by `gamma`. If `FALSE`, `gamma` is not used.
#' When the next layer is linear (also e.g. `relu`), this can be
#' disabled since the scaling will be done by the next layer.
#' Defaults to `TRUE`.
#'
#' @param beta_initializer
#' Initializer for the beta weight. Defaults to zeros.
#'
#' @param gamma_initializer
#' Initializer for the gamma weight. Defaults to ones.
#'
#' @param beta_regularizer
#' Optional regularizer for the beta weight. `NULL` by
#' default.
#'
#' @param gamma_regularizer
#' Optional regularizer for the gamma weight. `NULL` by
#' default.
#'
#' @param beta_constraint
#' Optional constraint for the beta weight.
#' `NULL` by default.
#'
#' @param gamma_constraint
#' Optional constraint for the gamma weight. `NULL` by
#' default.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit layer_dense return
#' @export
#' @family normalization layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/normalization_layers/group_normalization#groupnormalization-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GroupNormalization>
#' @tether keras.layers.GroupNormalization
layer_group_normalization <-
function (object, groups = 32L, axis = -1L, epsilon = 0.001,
    center = TRUE, scale = TRUE, beta_initializer = "zeros",
    gamma_initializer = "ones", beta_regularizer = NULL, gamma_regularizer = NULL,
    beta_constraint = NULL, gamma_constraint = NULL, ...)
{
    args <- capture_args(list(groups = as_integer, axis = as_axis,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$GroupNormalization, object, args)
}


#' Layer normalization layer (Ba et al., 2016).
#'
#' @description
#' Normalize the activations of the previous layer for each given example in a
#' batch independently, rather than across a batch like Batch Normalization.
#' i.e. applies a transformation that maintains the mean activation within each
#' example close to 0 and the activation standard deviation close to 1.
#'
#' If `scale` or `center` are enabled, the layer will scale the normalized
#' outputs by broadcasting them with a trainable variable `gamma`, and center
#' the outputs by broadcasting with a trainable variable `beta`. `gamma` will
#' default to a ones tensor and `beta` will default to a zeros tensor, so that
#' centering and scaling are no-ops before training has begun.
#'
#' So, with scaling and centering enabled the normalization equations
#' are as follows:
#'
#' Let the intermediate activations for a mini-batch to be the `inputs`.
#'
#' For each sample `x` in a batch of `inputs`, we compute the mean and
#' variance of the sample, normalize each value in the sample
#' (including a small factor `epsilon` for numerical stability),
#' and finally,
#' transform the normalized output by `gamma` and `beta`,
#' which are learned parameters:
#'
#' ```{r, eval = FALSE}
#' outputs <- inputs |> apply(1, function(x) {
#'   x_normalized <- (x - mean(x)) /
#'                   sqrt(var(x) + epsilon)
#'   x_normalized * gamma + beta
#' })
#' ```
#'
#' `gamma` and `beta` will span the axes of `inputs` specified in `axis`, and
#' this part of the inputs' shape must be fully defined.
#'
#' For example:
#'
#' ```{r}
#' layer <- layer_layer_normalization(axis = c(2, 3, 4))
#'
#' layer(op_ones(c(5, 20, 30, 40))) |> invisible() # build()
#' shape(layer$beta)
#' shape(layer$gamma)
#' ```
#'
#' Note that other implementations of layer normalization may choose to define
#' `gamma` and `beta` over a separate set of axes from the axes being
#' normalized across. For example, Group Normalization
#' ([Wu et al. 2018](https://arxiv.org/abs/1803.08494)) with group size of 1
#' corresponds to a `layer_layer_normalization()` that normalizes across height, width,
#' and channel and has `gamma` and `beta` span only the channel dimension.
#' So, this `layer_layer_normalization()` implementation will not match a
#' `layer_group_normalization()` layer with group size set to 1.
#'
#' # Reference
#' - [Lei Ba et al., 2016](https://arxiv.org/abs/1607.06450).
#'
#' @param axis
#' Integer or list. The axis or axes to normalize across.
#' Typically, this is the features axis/axes. The left-out axes are
#' typically the batch axis/axes. `-1` is the last dimension in the
#' input. Defaults to `-1`.
#'
#' @param epsilon
#' Small float added to variance to avoid dividing by zero.
#' Defaults to 1e-3.
#'
#' @param center
#' If `TRUE`, add offset of `beta` to normalized tensor. If `FALSE`,
#' `beta` is ignored. Defaults to `TRUE`.
#'
#' @param scale
#' If `TRUE`, multiply by `gamma`. If `FALSE`, `gamma` is not used.
#' When the next layer is linear (also e.g. `layer_activation_relu()`), this can be
#' disabled since the scaling will be done by the next layer.
#' Defaults to `TRUE`.
#'
#' @param rms_scaling
#' If `TRUE`, `center` and `scale` are ignored, and the
#' inputs are scaled by `gamma` and the inverse square root
#' of the square of all inputs. This is an approximate and faster
#' approach that avoids ever computing the mean of the input.
#'
#' @param beta_initializer
#' Initializer for the beta weight. Defaults to zeros.
#'
#' @param gamma_initializer
#' Initializer for the gamma weight. Defaults to ones.
#'
#' @param beta_regularizer
#' Optional regularizer for the beta weight.
#' `NULL` by default.
#'
#' @param gamma_regularizer
#' Optional regularizer for the gamma weight.
#' `NULL` by default.
#'
#' @param beta_constraint
#' Optional constraint for the beta weight.
#' `NULL` by default.
#'
#' @param gamma_constraint
#' Optional constraint for the gamma weight.
#' `NULL` by default.
#'
#' @param ...
#' Base layer keyword arguments (e.g. `name` and `dtype`).
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @inherit layer_dense return
#' @export
#' @family normalization layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/normalization_layers/layer_normalization#layernormalization-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization>
#' @tether keras.layers.LayerNormalization
layer_layer_normalization <-
function (object, axis = -1L, epsilon = 0.001, center = TRUE,
    scale = TRUE, rms_scaling = FALSE, beta_initializer = "zeros",
    gamma_initializer = "ones", beta_regularizer = NULL, gamma_regularizer = NULL,
    beta_constraint = NULL, gamma_constraint = NULL, ...)
{
    args <- capture_args(list(axis = as_axis, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$LayerNormalization, object, args)
}


#' Performs spectral normalization on the weights of a target layer.
#'
#' @description
#' This wrapper controls the Lipschitz constant of the weights of a layer by
#' constraining their spectral norm, which can stabilize the training of GANs.
#'
#' # Examples
#' Wrap `layer_conv_2d`:
#' ```{r}
#' x <- random_uniform(c(1, 10, 10, 1))
#' conv2d <- layer_spectral_normalization(
#'   layer = layer_conv_2d(filters = 2, kernel_size = 2)
#' )
#' y <- conv2d(x)
#' shape(y)
#' ```
#'
#' Wrap `layer_dense`:
#' ```{r}
#' x <- random_uniform(c(1, 10, 10, 1))
#' dense <- layer_spectral_normalization(layer = layer_dense(units = 10))
#' y <- dense(x)
#' shape(y)
#' ```
#'
#' # Reference
#' - [Spectral Normalization for GAN](https://arxiv.org/abs/1802.05957).
#'
#' @param layer
#' A `Layer` instance that
#' has either a `kernel` (e.g. `layer_conv_2d`, `layer_dense`...)
#' or an `embeddings` attribute (`layer_embedding` layer).
#'
#' @param power_iterations
#' int, the number of iterations during normalization.
#'
#' @param ...
#' Base wrapper keyword arguments.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @inherit layer_dense return
#' @export
#' @family normalization layers
#' @family layers
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/SpectralNormalization>
#'
#' @tether keras.layers.SpectralNormalization
layer_spectral_normalization <-
function (object, layer, power_iterations = 1L, ...)
{
    args <- capture_args(list(power_iterations = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$SpectralNormalization, object,
        args)
}


#' Unit normalization layer.
#'
#' @description
#' Normalize a batch of inputs so that each input in the batch has a L2 norm
#' equal to 1 (across the axes specified in `axis`).
#'
#' # Examples
#' ```{r}
#' data <- op_reshape(1:6, new_shape = c(2, 3))
#' normalized_data <- layer_unit_normalization(data)
#' op_sum(normalized_data[1,]^2)
#' ```
#'
#' @param axis
#' Integer or list. The axis or axes to normalize across.
#' Typically, this is the features axis or axes. The left-out axes are
#' typically the batch axis or axes. `-1` is the last dimension
#' in the input. Defaults to `-1`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit layer_dense return
#' @export
#' @family normalization layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/normalization_layers/unit_normalization#unitnormalization-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/UnitNormalization>
#'
#' @tether keras.layers.UnitNormalization
layer_unit_normalization <-
function (object, axis = -1L, ...)
{
    args <- capture_args(list(axis = as_axis, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$UnitNormalization, object, args)
}
