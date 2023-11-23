

#' Layer that applies an update to the cost function based input activity.
#'
#' @description
#'
#' # Input Shape
#' Arbitrary. Use the keyword argument `input_shape`
#' (tuple of integers, does not include the samples axis)
#' when using this layer as the first layer in a model.
#'
#' # Output Shape
#'     Same shape as input.
#'
#' @param l1
#' L1 regularization factor (positive float).
#'
#' @param l2
#' L2 regularization factor (positive float).
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family regularization layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/regularization_layers/activity_regularization#activityregularization-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/ActivityRegularization>
#' @tether keras.layers.ActivityRegularization
layer_activity_regularization <-
function (object, l1 = 0, l2 = 0, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$ActivityRegularization, object,
        args)
}


#' Applies dropout to the input.
#'
#' @description
#' The `Dropout` layer randomly sets input units to 0 with a frequency of
#' `rate` at each step during training time, which helps prevent overfitting.
#' Inputs not set to 0 are scaled up by `1 / (1 - rate)` such that the sum over
#' all inputs is unchanged.
#'
#' Note that the `Dropout` layer only applies when `training` is set to `TRUE`
#' in `call()`, such that no values are dropped during inference.
#' When using `model.fit`, `training` will be appropriately set to `TRUE`
#' automatically. In other contexts, you can set the argument explicitly
#' to `TRUE` when calling the layer.
#'
#' (This is in contrast to setting `trainable=FALSE` for a `Dropout` layer.
#' `trainable` does not affect the layer's behavior, as `Dropout` does
#' not have any variables/weights that can be frozen during training.)
#'
#' # Call Arguments
#' - `inputs`: Input tensor (of any rank).
#' - `training`: Python boolean indicating whether the layer should behave in
#'     training mode (adding dropout) or in inference mode (doing nothing).
#'
#' @param rate
#' Float between 0 and 1. Fraction of the input units to drop.
#'
#' @param noise_shape
#' 1D integer tensor representing the shape of the
#' binary dropout mask that will be multiplied with the input.
#' For instance, if your inputs have shape
#' `(batch_size, timesteps, features)` and
#' you want the dropout mask to be the same for all timesteps,
#' you can use `noise_shape=(batch_size, 1, features)`.
#'
#' @param seed
#' A Python integer to use as random seed.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family regularization layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/regularization_layers/dropout#dropout-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout>
#' @tether keras.layers.Dropout
layer_dropout <-
function (object, rate, noise_shape = NULL, seed = NULL, ...)
{
    args <- capture_args2(list(noise_shape = as_integer, seed = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$Dropout, object, args)
}


#' Apply multiplicative 1-centered Gaussian noise.
#'
#' @description
#' As it is a regularization layer, it is only active at training time.
#'
#' # Call Arguments
#' - `inputs`: Input tensor (of any rank).
#' - `training`: Python boolean indicating whether the layer should behave in
#'     training mode (adding dropout) or in inference mode (doing nothing).
#'
#' @param rate
#' Float, drop probability (as with `Dropout`).
#' The multiplicative noise will have
#' standard deviation `sqrt(rate / (1 - rate))`.
#'
#' @param seed
#' Integer, optional random seed to enable deterministic behavior.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family regularization layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/regularization_layers/gaussian_dropout#gaussiandropout-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GaussianDropout>
#' @tether keras.layers.GaussianDropout
layer_gaussian_dropout <-
function (object, rate, seed = NULL, ...)
{
    args <- capture_args2(list(seed = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$GaussianDropout, object, args)
}


#' Apply additive zero-centered Gaussian noise.
#'
#' @description
#' This is useful to mitigate overfitting
#' (you could see it as a form of random data augmentation).
#' Gaussian Noise (GS) is a natural choice as corruption process
#' for real valued inputs.
#'
#' As it is a regularization layer, it is only active at training time.
#'
#' # Call Arguments
#' - `inputs`: Input tensor (of any rank).
#' - `training`: Python boolean indicating whether the layer should behave in
#'     training mode (adding noise) or in inference mode (doing nothing).
#'
#' @param stddev
#' Float, standard deviation of the noise distribution.
#'
#' @param seed
#' Integer, optional random seed to enable deterministic behavior.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @export
#' @family regularization layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/regularization_layers/gaussian_noise#gaussiannoise-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/GaussianNoise>
#' @tether keras.layers.GaussianNoise
layer_gaussian_noise <-
function (object, stddev, seed = NULL, ...)
{
    args <- capture_args2(list(seed = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$GaussianNoise, object, args)
}


#' Spatial 1D version of Dropout.
#'
#' @description
#' This layer performs the same function as Dropout, however, it drops
#' entire 1D feature maps instead of individual elements. If adjacent frames
#' within feature maps are strongly correlated (as is normally the case in
#' early convolution layers) then regular dropout will not regularize the
#' activations and will otherwise just result in an effective learning rate
#' decrease. In this case, `SpatialDropout1D` will help promote independence
#' between feature maps and should be used instead.
#'
#' # Call Arguments
#' - `inputs`: A 3D tensor.
#' - `training`: Python boolean indicating whether the layer
#'     should behave in training mode (applying dropout)
#'     or in inference mode (pass-through).
#'
#' # Input Shape
#' 3D tensor with shape: `(samples, timesteps, channels)`
#'
#' # Output Shape
#' Same as input.
#'
#' # Reference
#' - [Tompson et al., 2014](https://arxiv.org/abs/1411.4280)
#'
#' @param rate
#' Float between 0 and 1. Fraction of the input units to drop.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param name
#' String, name for the object
#'
#' @param seed
#' Initial seed for the random number generator
#'
#' @param dtype
#' datatype (e.g., `"float32"`).
#'
#' @export
#' @family spatial dropout regularization layers
#' @family regularization layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/regularization_layers/spatial_dropout1d#spatialdropout1d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/SpatialDropout1D>
#' @tether keras.layers.SpatialDropout1D
layer_spatial_dropout_1d <-
function (object, rate, seed = NULL, name = NULL, dtype = NULL)
{
    args <- capture_args2(list(seed = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$SpatialDropout1D, object, args)
}


#' Spatial 2D version of Dropout.
#'
#' @description
#' This version performs the same function as Dropout, however, it drops
#' entire 2D feature maps instead of individual elements. If adjacent pixels
#' within feature maps are strongly correlated (as is normally the case in
#' early convolution layers) then regular dropout will not regularize the
#' activations and will otherwise just result in an effective learning rate
#' decrease. In this case, `SpatialDropout2D` will help promote independence
#' between feature maps and should be used instead.
#'
#' # Call Arguments
#' - `inputs`: A 4D tensor.
#' - `training`: Python boolean indicating whether the layer
#'     should behave in training mode (applying dropout)
#'     or in inference mode (pass-through).
#'
#' # Input Shape
#' 4D tensor with shape: `(samples, channels, rows, cols)` if
#'     data_format='channels_first'
#' or 4D tensor with shape: `(samples, rows, cols, channels)` if
#'     data_format='channels_last'.
#'
#' # Output Shape
#' Same as input.
#'
#' # Reference
#' - [Tompson et al., 2014](https://arxiv.org/abs/1411.4280)
#'
#' @param rate
#' Float between 0 and 1. Fraction of the input units to drop.
#'
#' @param data_format
#' `"channels_first"` or `"channels_last"`.
#' In `"channels_first"` mode, the channels dimension (the depth)
#' is at index 1, in `"channels_last"` mode is it at index 3.
#' It defaults to the `image_data_format` value found in your
#' Keras config file at `~/.keras/keras.json`.
#' If you never set it, then it will be `"channels_last"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param name
#' String, name for the object
#'
#' @param seed
#' Initial seed for the random number generator
#'
#' @param dtype
#' datatype (e.g., `"float32"`).
#'
#' @export
#' @family spatial dropout regularization layers
#' @family regularization layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/regularization_layers/spatial_dropout2d#spatialdropout2d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/SpatialDropout2D>
#' @tether keras.layers.SpatialDropout2D
layer_spatial_dropout_2d <-
function (object, rate, data_format = NULL, seed = NULL, name = NULL,
    dtype = NULL)
{
    args <- capture_args2(list(seed = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$SpatialDropout2D, object, args)
}


#' Spatial 3D version of Dropout.
#'
#' @description
#' This version performs the same function as Dropout, however, it drops
#' entire 3D feature maps instead of individual elements. If adjacent voxels
#' within feature maps are strongly correlated (as is normally the case in
#' early convolution layers) then regular dropout will not regularize the
#' activations and will otherwise just result in an effective learning rate
#' decrease. In this case, SpatialDropout3D will help promote independence
#' between feature maps and should be used instead.
#'
#' # Call Arguments
#' - `inputs`: A 5D tensor.
#' - `training`: Python boolean indicating whether the layer
#'         should behave in training mode (applying dropout)
#'         or in inference mode (pass-through).
#'
#' # Input Shape
#' 5D tensor with shape: `(samples, channels, dim1, dim2, dim3)` if
#'     data_format='channels_first'
#' or 5D tensor with shape: `(samples, dim1, dim2, dim3, channels)` if
#'     data_format='channels_last'.
#'
#' # Output Shape
#' Same as input.
#'
#' # Reference
#' - [Tompson et al., 2014](https://arxiv.org/abs/1411.4280)
#'
#' @param rate
#' Float between 0 and 1. Fraction of the input units to drop.
#'
#' @param data_format
#' `"channels_first"` or `"channels_last"`.
#' In `"channels_first"` mode, the channels dimension (the depth)
#' is at index 1, in `"channels_last"` mode is it at index 4.
#' It defaults to the `image_data_format` value found in your
#' Keras config file at `~/.keras/keras.json`.
#' If you never set it, then it will be `"channels_last"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param name
#' String, name for the object
#'
#' @param seed
#' Initial seed for the random number generator
#'
#' @param dtype
#' datatype (e.g., `"float32"`).
#'
#' @export
#' @family spatial dropout regularization layers
#' @family regularization layers
#' @family layers
#' @seealso
#' + <https:/keras.io/api/layers/regularization_layers/spatial_dropout3d#spatialdropout3d-class>
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/SpatialDropout3D>
#' @tether keras.layers.SpatialDropout3D
layer_spatial_dropout_3d <-
function (object, rate, data_format = NULL, seed = NULL, name = NULL,
    dtype = NULL)
{
    args <- capture_args2(list(seed = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$SpatialDropout3D, object, args)
}
