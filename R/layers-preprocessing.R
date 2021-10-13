
## ---- image preprocessing ----

#' Image resizing layer
#'
#' @details
#' Resize the batched image input to target height and width. The input should
#' be a 4D (batched) or 3D (unbatched) tensor in `"channels_last"` format.
#'
#' @inheritParams layer_dense
#'
#' @param height Integer, the height of the output shape.
#'
#' @param width Integer, the width of the output shape.
#'
#' @param interpolation String, the interpolation method. Defaults to `"bilinear"`.
#' Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`, `"lanczos3"`,
#' `"lanczos5"`, `"gaussian"`, and `"mitchellcubic"`.
#'
#' @param crop_to_aspect_ratio If TRUE, resize the images without aspect
#' ratio distortion. When the original aspect ratio differs from the target
#' aspect ratio, the output image will be cropped so as to return the largest
#' possible window in the image (of size `(height, width)`) that matches
#' the target aspect ratio. By default (`crop_to_aspect_ratio = FALSE`),
#' aspect ratio may not be preserved.
#'
#' @param ... standard layer arguments.
#'
#' @family image preprocessing layers
#' @family preprocessing layers
#'
#' @seealso
#'   -  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Resizing>
#'   -  <https://keras.io/api/layers/preprocessing_layers/image_preprocessing/resizing>
#' @export
layer_resizing <-
function(object, height, width, interpolation = "bilinear",
         crop_to_aspect_ratio = FALSE, ...)
{
  require_tf_version("2.6", "layer_resizing()")
  args <- capture_args(match.call(),
                       list(height = as.integer, width = as.integer,
                            interpolation = fix_string),
                       ignore = "object")
  create_layer(keras$layers$Resizing, object, args)
}


#' Multiply inputs by `scale` and adds `offset`
#'
#' @details
#' For instance:
#'
#' 1. To rescale an input in the `[0, 255]` range
#' to be in the `[0, 1]` range, you would pass `scale=1./255`.
#'
#' 2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` range,
#' you would pass `scale = 1/127.5, offset = -1`.
#'
#' The rescaling is applied both during training and inference.
#'
#' Input shape:
#'   Arbitrary.
#'
#' Output shape:
#'   Same as input.
#'
#' @inheritParams layer_dense
#'
#' @param scale Float, the scale to apply to the inputs.
#'
#' @param offset Float, the offset to apply to the inputs.
#' @param ... standard layer arguments.
#'
#' @family image preprocessing layers
#' @family preprocessing layers
#'
#' @seealso
#'   -  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling>
#'   -  <https://keras.io/api/layers/preprocessing_layers/image_preprocessing/rescaling>
#' @export
layer_rescaling <-
function(object, scale, offset = 0, ...)
{
  require_tf_version("2.6", "layer_rescaling()")
  args <- capture_args(match.call(), ignore = "object")
  create_layer(keras$layers$Rescaling, object, args)
}




#' Crop the central portion of the images to target height and width
#'
#' @details
#' Input shape:
#'   3D (unbatched) or 4D (batched) tensor with shape:
#'   `(..., height, width, channels)`, in `"channels_last"` format.
#'
#' Output shape:
#'   3D (unbatched) or 4D (batched) tensor with shape:
#'   `(..., target_height, target_width, channels)`.
#'
#' If the input height/width is even and the target height/width is odd (or
#' inversely), the input image is left-padded by 1 pixel.
#'
#' @inheritParams layer_dense
#'
#' @param height Integer, the height of the output shape.
#'
#' @param width Integer, the width of the output shape.
#'
#' @param ... standard layer arguments.
#'
#'
#' @family image preprocessing layers
#' @family preprocessing layers
#'
#' @seealso
#'   -  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/CenterCrop>
#'   -  <https://keras.io/api/layers/preprocessing_layers/image_preprocessing/center_crop>
#' @export
layer_center_crop <-
function(object, height, width, ...)
{
  require_tf_version("2.6", "layer_center_crop()")
  args <- capture_args(match.call(),
                       list(height = as.integer, width = as.integer),
                       ignore = "object")
  create_layer(keras$layers$CenterCrop, object, args)
}


## ---- image augmentation ----

#' Randomly crop the images to target height and width
#'
#' @details
#' This layer will crop all the images in the same batch to the same cropping
#' location.
#' By default, random cropping is only applied during training. At inference
#' time, the images will be first rescaled to preserve the shorter side, and
#' center cropped. If you need to apply random cropping at inference time,
#' set `training` to `TRUE` when calling the layer.
#'
#' Input shape:
#'   3D (unbatched) or 4D (batched) tensor with shape:
#'   `(..., height, width, channels)`, in `"channels_last"` format.
#'
#' Output shape:
#'   3D (unbatched) or 4D (batched) tensor with shape:
#'   `(..., target_height, target_width, channels)`.
#'
#' @inheritParams layer_dense
#'
#' @param height Integer, the height of the output shape.
#'
#' @param width Integer, the width of the output shape.
#'
#' @param seed Integer. Used to create a random seed.
#'
#' @param ... standard layer arguments.
#'
#' @family image augmentation layers
#' @family preprocessing layers
#'
#' @seealso
#'   -  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomCrop>
#'   -  <https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_crop>
#' @export
layer_random_crop <-
function(object, height, width, seed = NULL, ...)
{
  require_tf_version("2.6", "layer_random_crop()")
  args <- capture_args(match.call(),
                       list(height = as.integer, width = as.integer,
                            seed = as_nullable_integer),
                       ignore = "object")
  create_layer(keras$layers$RandomCrop, object, args)
}



#' Randomly flip each image horizontally and vertically
#'
#' @details
#' This layer will flip the images based on the `mode` attribute.
#' During inference time, the output will be identical to input. Call the layer
#' with `training = TRUE` to flip the input.
#'
#' Input shape:
#'   3D (unbatched) or 4D (batched) tensor with shape:
#'   `(..., height, width, channels)`, in `"channels_last"` format.
#'
#' Output shape:
#'   3D (unbatched) or 4D (batched) tensor with shape:
#'   `(..., height, width, channels)`, in `"channels_last"` format.
#'
#' @inheritParams layer_dense
#'
#' @param mode String indicating which flip mode to use. Can be `"horizontal"`,
#' `"vertical"`, or `"horizontal_and_vertical"`. Defaults to
#' `"horizontal_and_vertical"`. `"horizontal"` is a left-right flip and
#' `"vertical"` is a top-bottom flip.
#'
#' @param seed Integer. Used to create a random seed.
#'
#' @param ... standard layer arguments.
#'
#' @family image augmentation layers
#' @family preprocessing layers
#'
#' @seealso
#'   -  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomFlip>
#'   -  <https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_flip>
#' @export
layer_random_flip <-
function(object, mode = "horizontal_and_vertical", seed = NULL, ...)
{
  require_tf_version("2.6", "layer_random_flip()")
  args <- capture_args(match.call(),
                       list(seed = as_nullable_integer,
                            mode = fix_string),
                       ignore = "object")
  create_layer(keras$layers$RandomFlip, object, args)
}


#' Randomly translate each image during training
#'
#' @inheritParams layer_dense
#'
#' @param height_factor a float represented as fraction of value, or a list of size
#' 2 representing lower and upper bound for shifting vertically. A negative
#' value means shifting image up, while a positive value means shifting image
#' down. When represented as a single positive float, this value is used for
#' both the upper and lower bound. For instance, `height_factor = c(-0.2, 0.3)`
#' results in an output shifted by a random amount in the range
#' `[-20%, +30%]`.
#' `height_factor = 0.2` results in an output height shifted by a random amount
#' in the range `[-20%, +20%]`.
#'
#' @param width_factor a float represented as fraction of value, or a list of size 2
#' representing lower and upper bound for shifting horizontally. A negative
#' value means shifting image left, while a positive value means shifting
#' image right. When represented as a single positive float, this value is
#' used for both the upper and lower bound. For instance,
#' `width_factor = c(-0.2, 0.3)` results in an output shifted left by 20%, and
#' shifted right by 30%. `width_factor = 0.2` results in an output height
#' shifted left or right by 20%.
#'
#' @param fill_mode Points outside the boundaries of the input are filled according
#' to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
#' - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
#'   reflecting about the edge of the last pixel.
#' - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
#'   filling all values beyond the edge with the same constant value k = 0.
#' - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
#'   wrapping around to the opposite edge.
#' - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by the
#'   nearest pixel.
#'
#' @param interpolation Interpolation mode. Supported values: `"nearest"`,
#' `"bilinear"`.
#'
#' @param seed Integer. Used to create a random seed.
#'
#' @param fill_value a float represents the value to be filled outside the boundaries
#' when `fill_mode="constant"`.
#'
#' @param ... standard layer arguments.
#'
#' @family image augmentation layers
#' @family preprocessing layers
#'
#' @seealso
#'   -  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomTranslation>
#'   -  <https://keras.io/api/layers/preprocessing_layers/>
#' @export
layer_random_translation <-
function(object, height_factor, width_factor, fill_mode = "reflect",
         interpolation = "bilinear", seed = NULL, fill_value = 0, ...)
{
  require_tf_version("2.6", "layer_random_translation()")
  args <- capture_args(match.call(),
                       list(seed = as_nullable_integer,
                            interpolation = fix_string,
                            fill_mode = fix_string),
                       ignore = "object")
  create_layer(keras$layers$RandomTranslation, object, args)
}


#' Randomly rotate each image
#'
#' @details
#' By default, random rotations are only applied during training.
#' At inference time, the layer does nothing. If you need to apply random
#' rotations at inference time, set `training` to TRUE when calling the layer.
#'
#' Input shape:
#'   3D (unbatched) or 4D (batched) tensor with shape:
#'   `(..., height, width, channels)`, in `"channels_last"` format
#'
#' Output shape:
#'   3D (unbatched) or 4D (batched) tensor with shape:
#'   `(..., height, width, channels)`, in `"channels_last"` format
#'
#' @inheritParams layer_dense
#'
#' @param factor a float represented as fraction of 2 Pi, or a list of size 2
#' representing lower and upper bound for rotating clockwise and
#' counter-clockwise. A positive values means rotating counter clock-wise,
#' while a negative value means clock-wise. When represented as a single
#' float, this value is used for both the upper and lower bound. For
#' instance, `factor = c(-0.2, 0.3)` results in an output rotation by a random
#' amount in the range `[-20% * 2pi, 30% * 2pi]`. `factor = 0.2` results in an
#' output rotating by a random amount in the range `[-20% * 2pi, 20% * 2pi]`.
#'
#' @param fill_mode Points outside the boundaries of the input are filled according
#' to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
#' - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
#'   reflecting about the edge of the last pixel.
#' - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
#'   filling all values beyond the edge with the same constant value k = 0.
#' - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
#'   wrapping around to the opposite edge.
#' - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by the
#'   nearest pixel.
#'
#' @param interpolation Interpolation mode. Supported values: `"nearest"`,
#' `"bilinear"`.
#'
#' @param seed Integer. Used to create a random seed.
#'
#' @param fill_value a float represents the value to be filled outside the boundaries
#' when `fill_mode="constant"`.
#'
#' @param ... standard layer arguments.
#'
#'
#' @family image augmentation layers
#' @family preprocessing layers
#'
#' @seealso
#'   -  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomRotation>
#'   -  <https://keras.io/api/layers/preprocessing_layers/>
#' @export
layer_random_rotation <-
function(object, factor, fill_mode = "reflect", interpolation = "bilinear",
         seed = NULL, fill_value = 0, ...)
{
  require_tf_version("2.6", "layer_random_rotation()")
  args <- capture_args(match.call(),
                       list(seed = as_nullable_integer,
                            interpolation = fix_string,
                            fill_mode = fix_string),
                       ignore = "object")
  create_layer(keras$layers$RandomRotation, object, args)
}


#' A preprocessing layer which randomly zooms images during training.
#'
#' This layer will randomly zoom in or out on each axis of an image
#' independently, filling empty space according to fill_mode.
#'
#' @inheritParams layer_dense
#'
#' @param height_factor a float represented as fraction of value, or a list of size
#' 2 representing lower and upper bound for zooming vertically. When
#' represented as a single float, this value is used for both the upper and
#' lower bound. A positive value means zooming out, while a negative value
#' means zooming in. For instance, `height_factor = c(0.2, 0.3)` result in an
#' output zoomed out by a random amount in the range `[+20%, +30%]`.
#' `height_factor = c(-0.3, -0.2)` result in an output zoomed in by a random
#' amount in the range `[+20%, +30%]`.
#'
#' @param width_factor a float represented as fraction of value, or a list of size 2
#' representing lower and upper bound for zooming horizontally. When
#' represented as a single float, this value is used for both the upper and
#' lower bound. For instance, `width_factor = c(0.2, 0.3)` result in an output
#' zooming out between 20% to 30%. `width_factor = c(-0.3, -0.2)` result in an
#' output zooming in between 20% to 30%. Defaults to `NULL`, i.e., zooming
#' vertical and horizontal directions by preserving the aspect ratio.
#'
#' @param fill_mode Points outside the boundaries of the input are filled according
#' to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
#' - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
#'   reflecting about the edge of the last pixel.
#' - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
#'   filling all values beyond the edge with the same constant value k = 0.
#' - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
#'   wrapping around to the opposite edge.
#' - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by the
#'   nearest pixel.
#'
#' @param interpolation Interpolation mode. Supported values: `"nearest"`,
#' `"bilinear"`.
#'
#' @param seed Integer. Used to create a random seed.
#'
#' @param fill_value a float represents the value to be filled outside the boundaries
#' when `fill_mode="constant"`.
#'
#' @param ... standard layer arguments.
#'
#' @family image augmentation layers
#' @family preprocessing layers
#'
#' @seealso
#'   -  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomZoom>
#'   -  <https://keras.io/api/layers/preprocessing_layers/>
#' @export
layer_random_zoom <-
function(object, height_factor, width_factor = NULL, fill_mode = "reflect",
         interpolation = "bilinear", seed = NULL, fill_value = 0, ...)
{
  require_tf_version("2.6", "layer_random_zoom()")
  args <- capture_args(match.call(),
                       list(seed = as_nullable_integer,
                            interpolation = fix_string,
                            fill_mode = fix_string),
                       ignore = "object")
  create_layer(keras$layers$RandomZoom, object, args)
}


#' Adjust the contrast of an image or images by a random factor
#'
#' @details
#' Contrast is adjusted independently for each channel of each image during
#' training.
#'
#' For each channel, this layer computes the mean of the image pixels in the
#' channel and then adjusts each component `x` of each pixel to
#' `(x - mean) * contrast_factor + mean`.
#'
#' Input shape:
#'   3D (unbatched) or 4D (batched) tensor with shape:
#'   `(..., height, width, channels)`, in `"channels_last"` format.
#'
#' Output shape:
#'   3D (unbatched) or 4D (batched) tensor with shape:
#'   `(..., height, width, channels)`, in `"channels_last"` format.
#'
#' @inheritParams layer_dense
#'
#' @param factor a positive float represented as fraction of value, or a list of
#' size 2 representing lower and upper bound. When represented as a single
#' float, lower = upper. The contrast factor will be randomly picked between
#' `[1.0 - lower, 1.0 + upper]`.
#'
#' @param seed Integer. Used to create a random seed.
#'
#' @param ... standard layer arguments.
#'
#' @family image augmentation layers
#' @family preprocessing layers
#'
#' @seealso
#'   -  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomContrast>
#'   -  <https://keras.io/api/layers/preprocessing_layers/>
#' @export
layer_random_contrast <-
function(object, factor, seed = NULL, ...)
{
  require_tf_version("2.6", "layer_random_contrast()")
  args <- capture_args(match.call(), list(seed = as_nullable_integer),
                       ignore = "object")
  create_layer(keras$layers$RandomContrast, object, args)
}


#' Randomly vary the height of a batch of images during training
#'
#' @details
#' Adjusts the height of a batch of images by a random factor. The input
#' should be a 3D (unbatched) or 4D (batched) tensor in the `"channels_last"`
#' image data format.
#'
#' By default, this layer is inactive during inference.
#'
#' @inheritParams layer_dense
#'
#' @param factor A positive float (fraction of original height), or a list of size 2
#' representing lower and upper bound for resizing vertically. When
#' represented as a single float, this value is used for both the upper and
#' lower bound. For instance, `factor = c(0.2, 0.3)` results in an output with
#' height changed by a random amount in the range `[20%, 30%]`.
#' `factor = c(-0.2, 0.3)` results in an output with height changed by a random
#' amount in the range `[-20%, +30%]`. `factor=0.2` results in an output with
#' height changed by a random amount in the range `[-20%, +20%]`.
#'
#' @param interpolation String, the interpolation method. Defaults to `"bilinear"`.
#' Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,
#' `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
#'
#' @param seed Integer. Used to create a random seed.
#'
#' @param ... standard layer arguments.
#'
#' @family image augmentation layers
#' @family preprocessing layers
#'
#' @seealso
#'   -  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomHeight>
#'   -  <https://keras.io/api/layers/preprocessing_layers/>
#' @export
layer_random_height <-
function(object, factor, interpolation = "bilinear", seed = NULL, ...)
{
  require_tf_version("2.6", "layer_random_height()")
  args <- capture_args(match.call(),
                       list(seed = as_nullable_integer,
                            interpolation = fix_string),
                       ignore = "object")
  create_layer(keras$layers$RandomHeight, object, args)
}


#' Randomly vary the width of a batch of images during training
#'
#' @details
#' Adjusts the width of a batch of images by a random factor. The input
#' should be a 3D (unbatched) or 4D (batched) tensor in the `"channels_last"`
#' image data format.
#'
#' By default, this layer is inactive during inference.
#'
#' @inheritParams layer_dense
#'
#' @param factor A positive float (fraction of original height), or a list of size 2
#' representing lower and upper bound for resizing vertically. When
#' represented as a single float, this value is used for both the upper and
#' lower bound. For instance, `factor = c(0.2, 0.3)` results in an output with
#' width changed by a random amount in the range `[20%, 30%]`. `factor=(-0.2,
#' 0.3)` results in an output with width changed by a random amount in the
#' range `[-20%, +30%]`. `factor = 0.2` results in an output with width changed
#' by a random amount in the range `[-20%, +20%]`.
#'
#' @param interpolation String, the interpolation method. Defaults to `bilinear`.
#' Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`, `"lanczos3"`,
#' `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
#'
#' @param seed Integer. Used to create a random seed.
#'
#' @param ... standard layer arguments.
#'
#' @family image augmentation layers
#' @family preprocessing layers
#'
#' @seealso
#'   -  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomWidth>
#'   -  <https://keras.io/api/layers/preprocessing_layers/>
#' @export
layer_random_width <-
function(object, factor, interpolation = "bilinear", seed = NULL, ...)
{
  require_tf_version("2.6", "layer_random_width()")
  args <- capture_args(match.call(),
                       list(seed = as_nullable_integer,
                            interpolation = fix_string),
                       ignore = "object")
  create_layer(keras$layers$RandomWidth, object, args)
}






















## ---- categorical features preprocessing ----


#' A preprocessing layer which encodes integer features.
#'
#' @description
#'
#' This layer provides options for condensing data into a categorical encoding
#' when the total number of tokens are known in advance. It accepts integer
#' values as inputs, and it outputs a dense or sparse representation of those
#' inputs. For integer inputs where the total number of tokens is not known, use
#' [`layer_integer_lookup()`] instead.
#'
#' @inheritParams layer_dense
#'
#' @param num_tokens The total number of tokens the layer should support. All
#'   inputs to the layer must integers in the range `0 <= value < num_tokens`,
#'   or an error will be thrown.
#'
#' @param output_mode Specification for the output of the layer. Defaults to
#'   `"multi_hot"`. Values can be `"one_hot"`, `"multi_hot"` or `"count"`,
#'   configuring the layer as follows:
#'
#'   - `"one_hot"`: Encodes each individual element in the input into an array
#'   of `num_tokens` size, containing a 1 at the element index. If the last
#'   dimension is size 1, will encode on that dimension. If the last dimension
#'   is not size 1, will append a new dimension for the encoded output.
#'
#'   - `"multi_hot"`: Encodes each sample in the input into a single array of
#'   `num_tokens` size, containing a 1 for each vocabulary term present in the
#'   sample. Treats the last dimension as the sample dimension, if input shape
#'   is `(..., sample_length)`, output shape will be `(..., num_tokens)`.
#'
#'   - `"count"`: Like `"multi_hot"`, but the int array contains a count of the
#'   number of times the token at that index appeared in the sample.
#'
#'   For all output modes, currently only output up to rank 2 is supported.
#'
#' @param sparse Boolean. If `TRUE`, returns a `SparseTensor` instead of a dense
#'   `Tensor`. Defaults to `FALSE`.
#'
#' @param ... standard layer arguments.
#'
#' @family categorical features preprocessing layers
#' @family preprocessing layers
#'
#' @seealso
#'   - <https://www.tensorflow.org/api_docs/python/tf/keras/layers/CategoryEncoding>
#'   - <https://keras.io/api/layers/preprocessing_layers/categorical/category_encoding/>
#'
#' @export
layer_category_encoding <-
function(object, num_tokens=NULL, output_mode = "multi_hot", sparse = FALSE, ...)
{
  require_tf_version("2.6", "layer_category_encoding()")
  args <- capture_args(match.call(),
                       list(num_tokens = as_nullable_integer,
                            output_mode = fix_string),
                       ignore = "object")
  create_layer(keras$layers$CategoryEncoding, object, args)
}


#' A preprocessing layer which hashes and bins categorical features.
#'
#' @details
#' This layer transforms single or multiple categorical inputs to hashed output.
#' It converts a sequence of int or string to a sequence of int. The stable hash
#' function uses `tensorflow::ops::Fingerprint` to produce the same output
#' consistently across all platforms.
#'
#' This layer uses [FarmHash64](https://github.com/google/farmhash) by default,
#' which provides a consistent hashed output across different platforms and is
#' stable across invocations, regardless of device and context, by mixing the
#' input bits thoroughly.
#'
#' If you want to obfuscate the hashed output, you can also pass a random `salt`
#' argument in the constructor. In that case, the layer will use the
#' [SipHash64](https://github.com/google/highwayhash) hash function, with
#' the `salt` value serving as additional input to the hash function.
#'
#' **Example (FarmHash64)**
#' ````r
#' layer <- layer_hashing(num_bins=3)
#' inp <- matrix(c('A', 'B', 'C', 'D', 'E'))
#' layer(inp)
#' # <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
#' #   array([[1],
#' #          [0],
#' #          [1],
#' #          [1],
#' #          [2]])>
#' ````
#'
#' **Example (FarmHash64) with a mask value**
#' ````r
#' layer <- layer_hashing(num_bins=3, mask_value='')
#' inp <- matrix(c('A', 'B', 'C', 'D', 'E'))
#' layer(inp)
#' # <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
#' #   array([[1],
#' #          [1],
#' #          [0],
#' #          [2],
#' #          [2]])>
#' ````
#'
#' **Example (SipHash64)**
#' ````r
#' layer <- layer_hashing(num_bins=3, salt=c(133, 137))
#' inp <- matrix(c('A', 'B', 'C', 'D', 'E'))
#' layer(inp)
#' # <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
#' #   array([[1],
#' #          [2],
#' #          [1],
#' #          [0],
#' #          [2]])>
#' ````
#'
#' **Example (Siphash64 with a single integer, same as `salt=[133, 133]`)**
#' ````r
#' layer <- layer_hashing(num_bins=3, salt=133)
#' inp <- matrix(c('A', 'B', 'C', 'D', 'E'))
#' layer(inp)
#' # <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
#' #   array([[0],
#' #          [0],
#' #          [2],
#' #          [1],
#' #          [0]])>
#' ````
#'
#' @inheritParams layer_dense
#'
#' @param num_bins Number of hash bins. Note that this includes the `mask_value` bin,
#' so the effective number of bins is `(num_bins - 1)` if `mask_value` is
#' set.
#'
#' @param mask_value A value that represents masked inputs, which are mapped to
#' index 0. Defaults to NULL, meaning no mask term will be added and the
#' hashing will start at index 0.
#'
#' @param salt A single unsigned integer or NULL.
#' If passed, the hash function used will be SipHash64, with these values
#' used as an additional input (known as a "salt" in cryptography).
#' These should be non-zero. Defaults to `NULL` (in that
#' case, the FarmHash64 hash function is used). It also supports
#' list of 2 unsigned integer numbers, see reference paper for details.
#'
#' @param ... standard layer arguments.
#'
#' @family categorical features preprocessing layers
#' @family preprocessing layers
#'
#' @seealso
#'   -  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Hashing>
#'   -  <https://keras.io/api/layers/preprocessing_layers/categorical/hashing/>
#' @export
layer_hashing <-
function(object, num_bins, mask_value = NULL, salt = NULL, ...)
{
  require_tf_version("2.6", "layer_hashing()")
  args <- capture_args(match.call(),
                       list(num_bins = as.integer,
                            salt = as_nullable_integer),
                       ignore = "object")
  create_layer(keras$layers$Hashing, object, args)
}



#' A preprocessing layer which maps integer features to contiguous ranges.
#'
#' @details
#' This layer maps a set of arbitrary integer input tokens into indexed
#' integer output via a table-based vocabulary lookup. The layer's output indices
#' will be contiguously arranged up to the maximum vocab size, even if the input
#' tokens are non-continguous or unbounded. The layer supports multiple options
#' for encoding the output via `output_mode`, and has optional support for
#' out-of-vocabulary (OOV) tokens and masking.
#'
#' The vocabulary for the layer can be supplied on construction or learned via
#' `adapt()`. During `adapt()`, the layer will analyze a data set, determine the
#' frequency of individual integer tokens, and create a vocabulary from them. If
#' the vocabulary is capped in size, the most frequent tokens will be used to
#' create the vocabulary and all others will be treated as OOV.
#'
#' There are two possible output modes for the layer.
#' When `output_mode` is `"int"`,
#' input integers are converted to their index in the vocabulary (an integer).
#' When `output_mode` is `"multi_hot"`, `"count"`, or `"tf_idf"`, input integers
#' are encoded into an array where each dimension corresponds to an element in
#' the vocabulary.
#'
#' The vocabulary for the layer must be either supplied on construction or
#' learned via `adapt()`. During `adapt()`, the layer will analyze a data set,
#' determine the frequency of individual integer tokens, and create a vocabulary
#' from them. If the vocabulary is capped in size, the most frequent tokens will
#' be used to create the vocabulary and all others will be treated as OOV.
#'
#' @inheritParams layer_dense
#'
#' @param max_tokens The maximum size of the vocabulary for this layer. If `NULL`,
#' there is no cap on the size of the vocabulary. Note that this size
#' includes the OOV and mask tokens. Default to `NULL.`
#'
#' @param num_oov_indices The number of out-of-vocabulary tokens to use. If this
#' value is more than 1, OOV inputs are modulated to determine their OOV
#' value. If this value is 0, OOV inputs will cause an error when calling the
#' layer. Defaults to 1.
#'
#' @param mask_token An integer token that represents masked inputs. When
#' `output_mode` is `"int"`, the token is included in vocabulary and mapped
#' to index 0. In other output modes, the token will not appear in the
#' vocabulary and instances of the mask token in the input will be dropped.
#' If set to `NULL`, no mask term will be added. Defaults to `NULL`.
#'
#' @param oov_token Only used when `invert` is `TRUE.` The token to return for OOV
#' indices. Defaults to -1.
#'
#' @param vocabulary Optional. Either an array of integers or a string path to a text
#' file. If passing an array, can pass a list, list, 1D numpy array, or 1D
#' tensor containing the integer vocabulary terms. If passing a file path, the
#' file should contain one line per term in the vocabulary. If this argument
#' is set, there is no need to `adapt` the layer.
#'
#' @param invert Only valid when `output_mode` is `"int"`. If `TRUE`, this layer will
#' map indices to vocabulary items instead of mapping vocabulary items to
#' indices. Default to `FALSE`.
#'
#' @param output_mode Specification for the output of the layer. Defaults to `"int"`.
#' Values can be `"int"`, `"one_hot"`, `"multi_hot"`, `"count"`, or
#' `"tf_idf"` configuring the layer as follows:
#'   - `"int"`: Return the vocabulary indices of the input tokens.
#'   - `"one_hot"`: Encodes each individual element in the input into an
#'     array the same size as the vocabulary, containing a 1 at the element
#'     index. If the last dimension is size 1, will encode on that dimension.
#'     If the last dimension is not size 1, will append a new dimension for
#'     the encoded output.
#'   - `"multi_hot"`: Encodes each sample in the input into a single array
#'     the same size as the vocabulary, containing a 1 for each vocabulary
#'     term present in the sample. Treats the last dimension as the sample
#'     dimension, if input shape is (..., sample_length), output shape will
#'     be (..., num_tokens).
#'   - `"count"`: As `"multi_hot"`, but the int array contains a count of the
#'     number of times the token at that index appeared in the sample.
#'   - `"tf_idf"`: As `"multi_hot"`, but the TF-IDF algorithm is applied to
#'     find the value in each token slot.
#' For `"int"` output, any shape of input and output is supported. For all
#' other output modes, currently only output up to rank 2 is supported.
#'
#' @param pad_to_max_tokens Only applicable when `output_mode` is `"multi_hot"`,
#' `"count"`, or `"tf_idf"`. If TRUE, the output will have its feature axis
#' padded to `max_tokens` even if the number of unique tokens in the
#' vocabulary is less than max_tokens, resulting in a tensor of shape
#' `[batch_size, max_tokens]` regardless of vocabulary size. Defaults to `FALSE`.
#'
#' @param sparse Boolean. Only applicable when `output_mode` is `"multi_hot"`,
#' `"count"`, or `"tf_idf"`. If `TRUE`, returns a `SparseTensor` instead of a
#' dense `Tensor`. Defaults to `FALSE`.
#'
#' @param ... standard layer arguments.
#'
#' @family categorical features preprocessing layers
#' @family preprocessing layers
#'
#' @seealso
#'   - [`adapt()`]
#'   -  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/IntegerLookup>
#'   -  <https://keras.io/api/layers/preprocessing_layers/categorical/integer_lookup>
#' @export
layer_integer_lookup <-
function(object,
         max_tokens = NULL,
         num_oov_indices = 1L,
         mask_token = NULL,
         oov_token = -1L,
         vocabulary = NULL,
         invert = FALSE,
         output_mode = 'int',
         sparse = FALSE,
         pad_to_max_tokens = FALSE,
         ...)
{
  require_tf_version("2.6", "layer_integer_lookup()")
  args <- capture_args(match.call(),
                       list(num_oov_indices = as.integer,
                            oov_token = as.integer,
                            output_mode = fix_string),
                       ignore = "object")
  create_layer(keras$layers$IntegerLookup, object, args)
}



#' A preprocessing layer which maps string features to integer indices.
#'
#' @details
#' This layer translates a set of arbitrary strings into integer output via a
#' table-based vocabulary lookup.
#'
#' The vocabulary for the layer must be either supplied on construction or
#' learned via `adapt()`. During `adapt()`, the layer will analyze a data set,
#' determine the frequency of individual strings tokens, and create a vocabulary
#' from them. If the vocabulary is capped in size, the most frequent tokens will
#' be used to create the vocabulary and all others will be treated as
#' out-of-vocabulary (OOV).
#'
#' There are two possible output modes for the layer.
#' When `output_mode` is `"int"`,
#' input strings are converted to their index in the vocabulary (an integer).
#' When `output_mode` is `"multi_hot"`, `"count"`, or `"tf_idf"`, input strings
#' are encoded into an array where each dimension corresponds to an element in
#' the vocabulary.
#'
#' The vocabulary can optionally contain a mask token as well as an OOV token
#' (which can optionally occupy multiple indices in the vocabulary, as set
#' by `num_oov_indices`).
#' The position of these tokens in the vocabulary is fixed. When `output_mode` is
#' `"int"`, the vocabulary will begin with the mask token (if set), followed by
#' OOV indices, followed by the rest of the vocabulary. When `output_mode` is
#' `"multi_hot"`, `"count"`, or `"tf_idf"` the vocabulary will begin with OOV
#' indices and instances of the mask token will be dropped.
#'
#' @inheritParams layer_dense
#'
#' @param max_tokens The maximum size of the vocabulary for this layer. If `NULL`,
#' there is no cap on the size of the vocabulary. Note that this size
#' includes the OOV and mask tokens. Default to `NULL.`
#'
#' @param num_oov_indices The number of out-of-vocabulary tokens to use. If this
#' value is more than 1, OOV inputs are hashed to determine their OOV value.
#' If this value is 0, OOV inputs will cause an error when calling the layer.
#' Defaults to 1.
#'
#' @param mask_token A token that represents masked inputs. When `output_mode` is
#' `"int"`, the token is included in vocabulary and mapped to index 0. In
#' other output modes, the token will not appear in the vocabulary and
#' instances of the mask token in the input will be dropped. If set to `NULL`,
#' no mask term will be added. Defaults to `NULL`.
#'
#' @param oov_token Only used when `invert` is TRUE. The token to return for OOV
#' indices. Defaults to `"[UNK]"`.
#'
#' @param vocabulary Optional. Either an array of strings or a string path to a text
#' file. If passing an array, can pass a list, list, 1D numpy array, or 1D
#' tensor containing the string vocabulary terms. If passing a file path, the
#' file should contain one line per term in the vocabulary. If this argument
#' is set, there is no need to `adapt` the layer.
#'
#' @param encoding String encoding. Default of `NULL` is equivalent to `"utf-8"`.
#'
#' @param invert Only valid when `output_mode` is `"int"`. If TRUE, this layer will
#' map indices to vocabulary items instead of mapping vocabulary items to
#' indices. Default to `FALSE`.
#'
#' @param output_mode Specification for the output of the layer. Defaults to `"int"`.
#' Values can be `"int"`, `"one_hot"`, `"multi_hot"`, `"count"`, or
#' `"tf_idf"` configuring the layer as follows:
#'   - `"int"`: Return the raw integer indices of the input tokens.
#'   - `"one_hot"`: Encodes each individual element in the input into an
#'     array the same size as the vocabulary, containing a 1 at the element
#'     index. If the last dimension is size 1, will encode on that dimension.
#'     If the last dimension is not size 1, will append a new dimension for
#'     the encoded output.
#'   - `"multi_hot"`: Encodes each sample in the input into a single array
#'     the same size as the vocabulary, containing a 1 for each vocabulary
#'     term present in the sample. Treats the last dimension as the sample
#'     dimension, if input shape is (..., sample_length), output shape will
#'     be (..., num_tokens).
#'   - `"count"`: As `"multi_hot"`, but the int array contains a count of the
#'     number of times the token at that index appeared in the sample.
#'   - `"tf_idf"`: As `"multi_hot"`, but the TF-IDF algorithm is applied to
#'     find the value in each token slot.
#' For `"int"` output, any shape of input and output is supported. For all
#' other output modes, currently only output up to rank 2 is supported.
#'
#' @param pad_to_max_tokens Only applicable when `output_mode` is `"multi_hot"`,
#' `"count"`, or `"tf_idf"`. If TRUE, the output will have its feature axis
#' padded to `max_tokens` even if the number of unique tokens in the
#' vocabulary is less than max_tokens, resulting in a tensor of shape
#' `[batch_size, max_tokens]` regardless of vocabulary size. Defaults to `FALSE`.
#'
#' @param sparse Boolean. Only applicable when `output_mode` is `"multi_hot"`,
#' `"count"`, or `"tf_idf"`. If TRUE, returns a `SparseTensor` instead of a
#' dense `Tensor`. Defaults to `FALSE`.
#'
#' @param ... standard layer arguments.
#'
#' @family categorical features preprocessing layers
#' @family preprocessing layers
#'
#' @seealso
#'   - [`adapt()`]
#'   -  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/StringLookup>
#'   -  <https://keras.io/api/layers/preprocessing_layers/categorical/string_lookup>
#'
#' @export
layer_string_lookup <-
function(object,
         max_tokens = NULL,
         num_oov_indices = 1L,
         mask_token = NULL,
         oov_token = '[UNK]',
         vocabulary = NULL,
         encoding = NULL,
         invert = FALSE,
         output_mode = 'int',
         sparse = FALSE,
         pad_to_max_tokens = FALSE,
         ...)
{
  require_tf_version("2.6", "layer_string_lookup()")
  args <- capture_args(match.call(),
                       list(num_oov_indices = as.integer,
                            output_mode = fix_string),
                       ignore = "object")
  create_layer(keras$layers$StringLookup, object, args)
}


## ---- numerical features preprocessing ----



#' A preprocessing layer which normalizes continuous features.
#'
#' @details
#' This layer will shift and scale inputs into a distribution centered around 0
#' with standard deviation 1. It accomplishes this by precomputing the mean and
#' variance of the data, and calling `(input - mean) / sqrt(var)` at runtime.
#'
#' The mean and variance values for the layer must be either supplied on
#' construction or learned via `adapt()`. `adapt()` will compute the mean and
#' variance of the data and store them as the layer's weights. `adapt()` should
#' be called before `fit()`, `evaluate()`, or `predict()`.
#'
#' @inheritParams layer_dense
#'
#' @param axis Integer, list of integers, or NULL. The axis or axes that should
#' have a separate mean and variance for each index in the shape. For
#' example, if shape is `(NULL, 5)` and `axis=1`, the layer will track 5
#' separate mean and variance values for the last axis. If `axis` is set to
#' `NULL`, the layer will normalize all elements in the input by a scalar
#' mean and variance. Defaults to -1, where the last axis of the input is
#' assumed to be a feature dimension and is normalized per index. Note that
#' in the specific case of batched scalar inputs where the only axis is the
#' batch axis, the default will normalize each index in the batch
#' separately. In this case, consider passing `axis = NULL`.
#'
#' @param mean The mean value(s) to use during normalization. The passed value(s)
#' will be broadcast to the shape of the kept axes above; if the value(s)
#' cannot be broadcast, an error will be raised when this layer's `build()`
#' method is called.
#'
#' @param variance The variance value(s) to use during normalization. The passed
#' value(s) will be broadcast to the shape of the kept axes above; if the
#' value(s) cannot be broadcast, an error will be raised when this layer's
#' `build()` method is called.
#'
#' @param ... standard layer arguments.
#'
#' @family numerical features preprocessing layers
#' @family preprocessing layers
#'
#' @seealso
#'   - [`adapt()`]
#'   -  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization>
#'   -  <https://keras.io/api/layers/preprocessing_layers/numerical/normalization>
#' @export
layer_normalization <-
function(object, axis = -1L, mean=NULL, variance=NULL, ...)
{
  require_tf_version("2.6", "layer_normalization()")
  args <- capture_args(match.call(), list(axis = as_axis),
                       ignore = "object")
  create_layer(keras$layers$Normalization, object, args)
}


#' A preprocessing layer which buckets continuous features by ranges.
#'
#' @details
#' This layer will place each element of its input data into one of several
#' contiguous ranges and output an integer index indicating which range each
#' element was placed in.
#'
#' Input shape:
#'   Any `tf.Tensor` or `tf.RaggedTensor` of dimension 2 or higher.
#'
#' Output shape:
#'   Same as input shape.
#'
#' @inheritParams layer_dense
#'
#' @param bin_boundaries A list of bin boundaries. The leftmost and rightmost bins
#' will always extend to `-Inf` and `Inf`, so `bin_boundaries = c(0., 1., 2.)`
#' generates bins `(-Inf, 0.)`, `[0., 1.)`, `[1., 2.)`, and `[2., +Inf)`. If
#' this option is set, `adapt` should not be called.
#'
#' @param num_bins The integer number of bins to compute. If this option is set,
#' `adapt` should be called to learn the bin boundaries.
#'
#' @param epsilon Error tolerance, typically a small fraction close to zero (e.g.
#' 0.01). Higher values of epsilon increase the quantile approximation, and
#' hence result in more unequal buckets, but could improve performance
#' and resource consumption.
#'
#' @param ... standard layer arguments.
#'
#' @family numerical features preprocessing layers
#' @family preprocessing layers
#'
#' @seealso
#'   - [`adapt()`]
#'   -  <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Discretization>
#'   -  <https://keras.io/api/layers/preprocessing_layers/numerical/discretization>
#' @export
layer_discretization <-
function(object, bin_boundaries=NULL, num_bins = NULL, epsilon = 0.01, ...)
{
  require_tf_version("2.6", "layer_discretization()")
  args <- capture_args(match.call(),
                       ignore = "object")
  create_layer(keras$layers$Discretization, object, args)
}


# TODO: all the layers should have consistently formatted sections:
# Input shape, Output shape


# ---- text preprocessing ----


#' A preprocessing layer which maps text features to integer sequences.
#'
#' @details
#' This layer has basic options for managing text in a Keras model. It
#' transforms a batch of strings (one example = one string) into either a list of
#' token indices (one example = 1D tensor of integer token indices) or a dense
#' representation (one example = 1D tensor of float values representing data
#' about the example's tokens).
#'
#' The vocabulary for the layer must be either supplied on construction or
#' learned via `adapt()`. When this layer is adapted, it will analyze the
#' dataset, determine the frequency of individual string values, and create a
#' vocabulary from them. This vocabulary can have unlimited size or be capped,
#' depending on the configuration options for this layer; if there are more
#' unique values in the input than the maximum vocabulary size, the most
#' frequent terms will be used to create the vocabulary.
#'
#' The processing of each example contains the following steps:
#'
#' 1. Standardize each example (usually lowercasing + punctuation stripping)
#' 2. Split each example into substrings (usually words)
#' 3. Recombine substrings into tokens (usually ngrams)
#' 4. Index tokens (associate a unique int value with each token)
#' 5. Transform each example using this index, either into a vector of ints or
#'    a dense float vector.
#'
#' Some notes on passing callables to customize splitting and normalization for
#' this layer:
#'
#' 1. Any callable can be passed to this Layer, but if you want to serialize
#'    this object you should only pass functions that are registered Keras
#'    serializables (see [`tf$keras$utils$register_keras_serializable`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/register_keras_serializable)
#'    for more details).
#' 2. When using a custom callable for `standardize`, the data received
#'    by the callable will be exactly as passed to this layer. The callable
#'    should return a tensor of the same shape as the input.
#' 3. When using a custom callable for `split`, the data received by the
#'    callable will have the 1st dimension squeezed out - instead of
#'    `matrix(c("string to split", "another string to split"))`, the Callable will
#'    see `c("string to split", "another string to split")`. The callable should
#'    return a Tensor with the first dimension containing the split tokens -
#'    in this example, we should see something like `list(c("string", "to",
#'    "split"), c("another", "string", "to", "split"))`. This makes the callable
#'    site natively compatible with `tf$strings$split()`.
#'
#' @inheritParams layer_dense
#'
#' @param max_tokens The maximum size of the vocabulary for this layer. If NULL,
#' there is no cap on the size of the vocabulary. Note that this vocabulary
#' contains 1 OOV token, so the effective number of tokens is `(max_tokens -
#' 1 - (1 if output_mode == "int" else 0))`.
#'
#' @param standardize Optional specification for standardization to apply to the
#' input text. Values can be NULL (no standardization),
#' `"lower_and_strip_punctuation"` (lowercase and remove punctuation) or a
#' Callable. Default is `"lower_and_strip_punctuation"`.
#'
#' @param split Optional specification for splitting the input text. Values can be
#' NULL (no splitting), `"whitespace"` (split on ASCII whitespace), or a
#' Callable. The default is `"whitespace"`.
#'
#' @param ngrams Optional specification for ngrams to create from the possibly-split
#' input text. Values can be NULL, an integer or list of integers; passing
#' an integer will create ngrams up to that integer, and passing a list of
#' integers will create ngrams for the specified values in the list. Passing
#' NULL means that no ngrams will be created.
#'
#' @param output_mode Optional specification for the output of the layer. Values can
#' be `"int"`, `"multi_hot"`, `"count"` or `"tf_idf"`, configuring the layer
#' as follows:
#'   - `"int"`: Outputs integer indices, one integer index per split string
#'     token. When `output_mode == "int"`, 0 is reserved for masked
#'     locations; this reduces the vocab size to
#'     `max_tokens - 2` instead of `max_tokens - 1`.
#'   - `"multi_hot"`: Outputs a single int array per batch, of either
#'     vocab_size or max_tokens size, containing 1s in all elements where the
#'     token mapped to that index exists at least once in the batch item.
#'   - `"count"`: Like `"multi_hot"`, but the int array contains a count of
#'     the number of times the token at that index appeared in the
#'     batch item.
#'   - `"tf_idf"`: Like `"multi_hot"`, but the TF-IDF algorithm is applied to
#'     find the value in each token slot.
#' For `"int"` output, any shape of input and output is supported. For all
#' other output modes, currently only rank 1 inputs (and rank 2 outputs after
#' splitting) are supported.
#'
#' @param output_sequence_length Only valid in INT mode. If set, the output will have
#' its time dimension padded or truncated to exactly `output_sequence_length`
#' values, resulting in a tensor of shape
#' `(batch_size, output_sequence_length)` regardless of how many tokens
#' resulted from the splitting step. Defaults to NULL.
#'
#' @param pad_to_max_tokens Only valid in  `"multi_hot"`, `"count"`, and `"tf_idf"`
#' modes. If TRUE, the output will have its feature axis padded to
#' `max_tokens` even if the number of unique tokens in the vocabulary is less
#' than max_tokens, resulting in a tensor of shape `(batch_size, max_tokens)`
#' regardless of vocabulary size. Defaults to FALSE.
#'
#' @param vocabulary Optional for `layer_text_vectorization()`. Either an array
#'   of strings or a string path to a text file. If passing an array, can pass
#'   an R list or character vector, 1D numpy array, or 1D tensor containing the
#'   string vocabulary terms. If passing a file path, the file should contain
#'   one line per term in the vocabulary. If vocabulary is set (either by
#'   passing `layer_text_vectorization(vocabulary = ...)` or by calling
#'   `set_vocabulary(layer, vocabulary = ...`), there is no need to `adapt()`
#'   the layer.
#'
#' @family text preprocessing layers
#' @family preprocessing layers
#'
#'
#' @param ... standard layer arguments.
#'
#' @seealso
#'   - [`adapt()`]
#'   - <https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization>
#'   - <https://keras.io/api/layers/preprocessing_layers/text/text_vectorization>
#' @export
layer_text_vectorization <-
function(object,
         max_tokens = NULL,
         standardize = 'lower_and_strip_punctuation',
         split = 'whitespace',
         ngrams = NULL,
         output_mode = 'int',
         output_sequence_length = NULL,
         pad_to_max_tokens = FALSE,
         vocabulary = NULL,
         ...)
{

  # TODO: in TF 2.8, new args: sparse=FALSE, ragged=FALSE, idf_weights=NULL
  if (tf_version() >= "2.6") {
    callable <- keras$layers$TextVectorization
    # output_mode_choices <- c("int", "multi_hot", "count", "tf_idf")
  } else {
    # warning("Defaults to layer_text_vectorization() were changed in Tensorflow 2.6.",
    #         "Please consult the docs and update your code")
    callable <- keras$layers$experimental$preprocessing$TextVectorization
    # output_mode_choices <- c("int", "binary", "count", "tf-idf")
    # on the python side, "binary" is renamed to "multi_hot", "tf-idf" renamed to "tf_idf"
  }


  modifiers <-  list(
    max_tokens = as_nullable_integer,
    output_sequence_length = as_nullable_integer,
    ngrams = function(x)
      if (length(x) > 1) as_integer_tuple(x) else as_nullable_integer(x),
    # output_mode = function(x) fix_string(match.arg(x, output_mode_choices)),
    output_mode = fix_string,
    split = fix_string,
    standardize = fix_string
  )

  args <- capture_args(match.call(), modifiers, ignore = "object")

  if("vocabulary" %in% names(args))
    require_tf_version("2.4", "pass `vocabulary` argument to layer_text_vectorization()")

  create_layer(callable, object, args)
}


#' @param include_special_tokens If True, the returned vocabulary will include
#'   the padding and OOV tokens, and a term's index in the vocabulary will equal
#'   the term's index when calling the layer. If False, the returned vocabulary
#'   will not include any padding or OOV tokens.
#' @rdname layer_text_vectorization
#' @export
get_vocabulary <- function(object, include_special_tokens=TRUE) {
  if (tensorflow::tf_version() < "2.3") {
    python_path <- system.file("python", package = "keras")
    tools <- import_from_path("kerastools", path = python_path)
    tools$get_vocabulary$get_vocabulary(object)
  } else {
    args <- capture_args(match.call(), ignore = "object")
    do.call(object$get_vocabulary, args)
  }
}

#' @rdname layer_text_vectorization
#' @param idf_weights An R vector, 1D numpy array, or 1D tensor of inverse
#'   document frequency weights with equal length to vocabulary. Must be set if
#'   output_mode is "tf_idf". Should not be set otherwise.
#' @export
set_vocabulary <- function(object, vocabulary, idf_weights=NULL, ...) {
  args <- capture_args(match.call(), ignore = "object")

  if (tf_version() < "2.6") {
    # required arg renamed when promoted out of experimental in 2.6:
    #   vocab -> vocabulary. Position as first unchanged.
    vocab <- args[["vocabulary"]] %||% args[["vocab"]]
    args[c("vocab", "vocabulary")] <- NULL
    args <- c(list(vocab), args)
  }

  do.call(object$set_vocabulary, args)
  invisible(object)
}



#' Fits the state of the preprocessing layer to the data being passed
#'
#' @details
#' After calling `adapt` on a layer, a preprocessing layer's state will not
#' update during training. In order to make preprocessing layers efficient in
#' any distribution context, they are kept constant with respect to any
#' compiled `tf.Graph`s that call the layer. This does not affect the layer use
#' when adapting each layer only once, but if you adapt a layer multiple times
#' you will need to take care to re-compile any compiled functions as follows:
#'
#'  * If you are adding a preprocessing layer to a `keras.Model`, you need to
#'    call `compile(model)` after each subsequent call to `adapt()`.
#'  * If you are calling a preprocessing layer inside `tfdatasets::dataset_map()`,
#'    you should call `dataset_map()` again on the input `tf.data.Dataset` after each
#'    `adapt()`.
#'  * If you are using a `tensorflow::tf_function()` directly which calls a preprocessing
#'    layer, you need to call `tf_function` again on your callable after
#'    each subsequent call to `adapt()`.
#'
#' `keras_model` example with multiple adapts:
#' ````r
#' layer <- layer_normalization(axis=NULL)
#' adapt(layer, c(0, 2))
#' model <- keras_model_sequential(layer)
#' predict(model, c(0, 1, 2)) # [1] -1  0  1
#'
#' adapt(layer, c(-1, 1))
#' compile(model)  # This is needed to re-compile model.predict!
#' predict(model, c(0, 1, 2)) # [1] 0 1 2
#' ````
#'
#' `tf.data.Dataset` example with multiple adapts:
#' ````r
#' layer <- layer_normalization(axis=NULL)
#' adapt(layer, c(0, 2))
#' input_ds <- tfdatasets::range_dataset(0, 3)
#' normalized_ds <- input_ds %>%
#'   tfdatasets::dataset_map(layer)
#' str(reticulate::iterate(normalized_ds))
#' # List of 3
#' #  $ :tf.Tensor([-1.], shape=(1,), dtype=float32)
#' #  $ :tf.Tensor([0.], shape=(1,), dtype=float32)
#' #  $ :tf.Tensor([1.], shape=(1,), dtype=float32)
#' adapt(layer, c(-1, 1))
#' normalized_ds <- input_ds %>%
#'   tfdatasets::dataset_map(layer) # Re-map over the input dataset.
#' str(reticulate::iterate(normalized_ds$as_numpy_iterator()))
#' # List of 3
#' #  $ : num [1(1d)] -1
#' #  $ : num [1(1d)] 0
#' #  $ : num [1(1d)] 1
#' ````
#'
#' @param object Preprocessing layer object
#'
#' @param data The data to train on. It can be passed either as a
#'   `tf.data.Dataset` or as an R array.
#'
#' @param batch_size Integer or `NULL`. Number of asamples per state update. If
#'   unspecified, `batch_size` will default to `32`. Do not specify the
#'   batch_size if your data is in the form of datasets, generators, or
#'   `keras.utils.Sequence` instances (since they generate batches).
#'
#' @param steps Integer or `NULL`. Total number of steps (batches of samples)
#'   When training with input tensors such as TensorFlow data tensors, the
#'   default `NULL` is equal to the number of samples in your dataset divided by
#'   the batch size, or `1` if that cannot be determined. If x is a
#'   `tf.data.Dataset`, and `steps` is `NULL`, the epoch will run until the
#'   input dataset is exhausted. When passing an infinitely repeating dataset,
#'   you must specify the steps argument. This argument is not supported with
#'   array inputs.
#'
#' @param ... Used for forwards and backwards compatibility. Passed on to the underlying method.
#'
#' @family preprocessing layer methods
#'
#' @seealso
#'
#'  + <https://www.tensorflow.org/guide/keras/preprocessing_layers#the_adapt_method>
#'  + <https://keras.io/guides/preprocessing_layers/#the-adapt-method>
#'
#' @export
adapt <- function(object, data, ..., batch_size=NULL, steps=NULL) {
  if (!inherits(data, "python.builtin.object"))
    data <- keras_array(data)
  # TODO: use as_tensor() here

  args <- capture_args(match.call(),
    list(batch_size = as_nullable_integer,
         step = as_nullable_integer),
    ignore = "object")
  do.call(object$adapt, args)
  invisible(object)
}





## TODO: TextVectorization has a compile() method. investigate if this is
## actually useful to export
#compile.keras.engine.base_preprocessing_layer.PreprocessingLayer <-
function(object, run_eagerly = NULL, steps_per_execution = NULL, ...) {
  args <- capture_args(match.call(), ignore="object")
  do.call(object$compile, args)
}



# TODO: add an 'experimental' tag in the R docs where appropriate

require_tf_version <- function(ver, msg = "this function.") {
  if (tf_version() < ver)
    stop("Tensorflow version >=", ver, " required to use ", msg)
}


# in python keras, sometimes strings are compared with `is`, which is too strict
# https://github.com/keras-team/keras/blob/db3fa5d40ed19cdf89fc295e8d0e317fb64480d4/keras/layers/preprocessing/text_vectorization.py#L524
# # there was already 1 PR submitted to fix this:
# https://github.com/tensorflow/tensorflow/pull/34420
# This is a hack around that: deparse and reparse the string in python. This
# gives the python interpreter a chance to recycle the address for the identical
# string that's already in it's string pool, which then passes the `is`
# comparison.
fix_string <- local({
  py_reparse <- NULL # R CMD check: no visible binding
  delayedAssign("py_reparse",
    py_eval("lambda x: eval(repr(x), {'__builtins__':{}}, {})",
            convert = FALSE))
  # Note, the globals dict {'__builtins__':{}} is a guardrail, not a security
  # door. A well crafted string can still break out to execute arbitrary code in
  # the python session, but it can do no more than can be done from the R
  # process already.
  # Can't use ast.literal_eval() because it fails the 'is' test. E.g.:
  # bool('foo' is ast.literal_eval("'foo'")) == False
  function(x) {
    if (is.character(x))
      py_call(py_reparse, as.character(x))
    else
      x
  }
})
