
#' Applies Dropout to the input.
#' 
#' Dropout consists in randomly setting a fraction `rate` of input units to 0 at
#' each update during training time, which helps prevent overfitting.
#' 
#' @inheritParams layer_dense
#' 
#' @param rate float between 0 and 1. Fraction of the input units to drop.
#' @param noise_shape 1D integer tensor representing the shape of the binary
#'   dropout mask that will be multiplied with the input. For instance, if your
#'   inputs have shape `(batch_size, timesteps, features)` and you want the
#'   dropout mask to be the same for all timesteps, you can use
#'   `noise_shape=c(batch_size, 1, features)`.
#' @param seed integer to use as random seed.
#'
#' @family core layers
#' @family dropout layers   
#'         
#' @export
layer_dropout <- function(object, rate, noise_shape = NULL, seed = NULL, 
                          input_shape = NULL, batch_input_shape = NULL,
                          batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {
  
  create_layer(keras$layers$Dropout, object, list(
    rate = rate,
    noise_shape = normalize_shape(noise_shape),
    seed = seed,
    input_shape = normalize_shape(input_shape),
    batch_input_shape = normalize_shape(batch_input_shape),
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
}

#' Spatial 1D version of Dropout.
#' 
#' This version performs the same function as Dropout, however it drops entire 
#' 1D feature maps instead of individual elements. If adjacent frames within 
#' feature maps are strongly correlated (as is normally the case in early 
#' convolution layers) then regular dropout will not regularize the activations 
#' and will otherwise just result in an effective learning rate decrease. In 
#' this case, `layer_spatial_dropout_1d` will help promote independence between 
#' feature maps and should be used instead.
#' 
#' @inheritParams layer_dropout
#'   
#' @section Input shape: 3D tensor with shape: `(samples, timesteps, channels)`
#'   
#' @section Output shape: Same as input
#'   
#' @section References: - [Efficient Object Localization Using Convolutional 
#'   Networks](https://arxiv.org/abs/1411.4280)
#' 
#' @family dropout layers
#'       
#' @export
layer_spatial_dropout_1d <- function(object, rate, 
                                     batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {
  
  create_layer(keras$layers$SpatialDropout1D, object, list(
    rate = rate,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
}

#' Spatial 2D version of Dropout.
#' 
#' This version performs the same function as Dropout, however it drops entire
#' 2D feature maps instead of individual elements. If adjacent pixels within
#' feature maps are strongly correlated (as is normally the case in early
#' convolution layers) then regular dropout will not regularize the activations
#' and will otherwise just result in an effective learning rate decrease. In
#' this case, `layer_spatial_dropout_2d` will help promote independence between
#' feature maps and should be used instead.
#' 
#' @inheritParams layer_spatial_dropout_1d
#' 
#' @param rate float between 0 and 1. Fraction of the input units to drop.
#' @param data_format 'channels_first' or 'channels_last'. In 'channels_first'
#'   mode, the channels dimension (the depth) is at index 1, in 'channels_last'
#'   mode is it at index 3. It defaults to the `image_data_format` value found
#'   in your Keras config file at `~/.keras/keras.json`. If you never set it,
#'   then it will be "channels_last".
#'   
#' @section Input shape: 4D tensor with shape: `(samples, channels, rows, cols)`
#'   if data_format='channels_first' or 4D tensor with shape: `(samples, rows,
#'   cols, channels)` if data_format='channels_last'.
#'   
#' @section Output shape: Same as input
#'   
#' @section References: - [Efficient Object Localization Using Convolutional
#'   Networks](https://arxiv.org/abs/1411.4280)
#'  
#' @family dropout layers 
#'   
#' @export
layer_spatial_dropout_2d <- function(object, rate, data_format = NULL, 
                                     batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {
  
  create_layer(keras$layers$SpatialDropout2D, object, list(
    rate = rate,
    data_format = data_format,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
}



#' Spatial 3D version of Dropout.
#' 
#' This version performs the same function as Dropout, however it drops entire
#' 3D feature maps instead of individual elements. If adjacent voxels within
#' feature maps are strongly correlated (as is normally the case in early
#' convolution layers) then regular dropout will not regularize the activations
#' and will otherwise just result in an effective learning rate decrease. In
#' this case, `layer_spatial_dropout_3d` will help promote independence between
#' feature maps and should be used instead.
#' 
#' @inheritParams layer_spatial_dropout_1d
#' 
#' @param data_format 'channels_first' or 'channels_last'. In 'channels_first'
#'   mode, the channels dimension (the depth) is at index 1, in 'channels_last'
#'   mode is it at index 4. It defaults to the `image_data_format` value found
#'   in your Keras config file at `~/.keras/keras.json`. If you never set it,
#'   then it will be "channels_last".
#'   
#' @section Input shape: 5D tensor with shape: `(samples, channels, dim1, dim2,
#'   dim3)` if data_format='channels_first' or 5D tensor with shape: `(samples,
#'   dim1, dim2, dim3, channels)` if data_format='channels_last'.
#'   
#' @section Output shape: Same as input
#'   
#' @section References: - [Efficient Object Localization Using Convolutional
#'   Networks](https://arxiv.org/abs/1411.4280)
#' 
#' @family dropout layers
#'       
#' @export
layer_spatial_dropout_3d <- function(object, rate, data_format = NULL, 
                                     batch_size = NULL, name = NULL, trainable = NULL, weights = NULL) {
  
  create_layer(keras$layers$SpatialDropout3D, object, list(
    rate = rate,
    data_format = data_format,
    batch_size = as_nullable_integer(batch_size),
    name = name,
    trainable = trainable,
    weights = weights
  ))
  
}

