

#' Applies the given transform(s) to the image(s).
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(2, 64, 80, 3)) # batch of 2 RGB images
#' transform <- op_array(rbind(c(1.5, 0, -20, 0, 1.5, -16, 0, 0),  # zoom
#'                            c(1, 0, -20, 0, 1, -16, 0, 0)))  # translation))
#' y <- op_image_affine_transform(x, transform)
#' shape(y)
#' # (2, 64, 80, 3)
#' ```
#'
#' ```{r}
#' x <- random_uniform(c(64, 80, 3)) # single RGB image
#' transform <- op_array(c(1.0, 0.5, -20, 0.5, 1.0, -16, 0, 0))  # shear
#' y <- op_image_affine_transform(x, transform)
#' shape(y)
#' # (64, 80, 3)
#' ```
#'
#' ```{r}
#' x <- random_uniform(c(2, 3, 64, 80)) # batch of 2 RGB images
#' transform <- op_array(rbind(
#'   c(1.5, 0,-20, 0, 1.5,-16, 0, 0),  # zoom
#'   c(1, 0,-20, 0, 1,-16, 0, 0)  # translation
#' ))
#' y <- op_image_affine_transform(x, transform, data_format = "channels_first")
#' shape(y)
#' # (2, 3, 64, 80)
#' ```
#'
#' @returns
#' Applied affine transform image or batch of images.
#'
#' @param images
#' Input image or batch of images. Must be 3D or 4D.
#'
#' @param transform
#' Projective transform matrix/matrices. A vector of length 8 or
#' tensor of size N x 8. If one row of transform is
#' `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps the output point
#' `(x, y)` to a transformed input point
#' `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
#' where `k = c0 x + c1 y + 1`. The transform is inverted compared to
#' the transform mapping input points to output points. Note that
#' gradients are not backpropagated into transformation parameters.
#' Note that `c0` and `c1` are only effective when using TensorFlow
#' backend and will be considered as `0` when using other backends.
#'
#' @param interpolation
#' Interpolation method. Available methods are `"nearest"`,
#' and `"bilinear"`. Defaults to `"bilinear"`.
#'
#' @param fill_mode
#' Points outside the boundaries of the input are filled
#' according to the given mode. Available methods are `"constant"`,
#' `"nearest"`, `"wrap"` and `"reflect"`. Defaults to `"constant"`.
#' - `"reflect"`: `(d c b a | a b c d | d c b a)`
#'     The input is extended by reflecting about the edge of the last
#'     pixel.
#' - `"constant"`: `(k k k k | a b c d | k k k k)`
#'     The input is extended by filling all values beyond
#'     the edge with the same constant value k specified by
#'     `fill_value`.
#' - `"wrap"`: `(a b c d | a b c d | a b c d)`
#'     The input is extended by wrapping around to the opposite edge.
#' - `"nearest"`: `(a a a a | a b c d | d d d d)`
#'     The input is extended by the nearest pixel.
#'
#' @param fill_value
#' Value used for points outside the boundaries of the input if
#' `fill_mode = "constant"`. Defaults to `0`.
#'
#' @param data_format
#' A string specifying the data format of the input tensor.
#' It can be either `"channels_last"` or `"channels_first"`.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch, height, width, channels)`, while `"channels_first"`
#' corresponds to inputs with shape `(batch, channels, height, width)`.
#'
#' If not specified, the value will default to
#' `config_image_data_format()`.
#'
#' @export
#' @family image ops
#' @family image utils
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/image#affinetransform-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/image/affine_transform>
#'
#' @tether keras.ops.image.affine_transform
op_image_affine_transform <-
function (images, transform, interpolation = "bilinear", fill_mode = "constant",
    fill_value = 0L, data_format = NULL)
{
    args <- capture_args(list(fill_value = as_integer))

    # pass 'images' as unnamed positional arg (was renamed from 'image' in Keras 3.4.0)
    args <- args_to_positional(args, "images")

    do.call(ops$image$affine_transform, args)
}


#' Extracts patches from the image(s).
#'
#' @description
#'
#' # Examples
#' ```{r}
#' image <- random_uniform(c(2, 20, 20, 3), dtype = "float32") # batch of 2 RGB images
#' patches <- op_image_extract_patches(image, c(5, 5))
#' shape(patches)
#' # (2, 4, 4, 75)
#' image <- random_uniform(c(20, 20, 3), dtype = "float32") # 1 RGB image
#' patches <- op_image_extract_patches(image, c(3, 3), c(1, 1))
#' shape(patches)
#' # (18, 18, 27)
#' ```
#'
#' @returns
#' Extracted patches 3D (if not batched) or 4D (if batched)
#'
#' @param images
#' Input image or batch of images. Must be 3D or 4D.
#'
#' @param size
#' Patch size int or list (patch_height, patch_width)
#'
#' @param strides
#' strides along height and width. If not specified, or
#' if `NULL`, it defaults to the same value as `size`.
#'
#' @param dilation_rate
#' This is the input stride, specifying how far two
#' consecutive patch samples are in the input. For value other than 1,
#' strides must be 1. NOTE: `strides > 1` is not supported in
#' conjunction with `dilation_rate > 1`
#'
#' @param padding
#' The type of padding algorithm to use: `"same"` or `"valid"`.
#'
#' @param data_format
#' A string specifying the data format of the input tensor.
#' It can be either `"channels_last"` or `"channels_first"`.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch, height, width, channels)`, while `"channels_first"`
#' corresponds to inputs with shape `(batch, channels, height, width)`.
#' If not specified, the value will default to
#' `config_image_data_format()`.
#'
#' @export
#' @family image ops
#' @family image utils
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/image#extractpatches-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/image/extract_patches>
#'
#' @tether keras.ops.image.extract_patches
op_image_extract_patches <-
function (images, size, strides = NULL, dilation_rate = 1L, padding = "valid",
    data_format = "channels_last")
{
    args <- capture_args(list(size = as_integer, dilation_rate = as_integer))
    # pass 'images' as unnamed positional arg (was renamed from 'image' in Keras 3.4.0)
    args <- args_to_positional(args, "images")

    do.call(ops$image$extract_patches, args)
}


#' Map the input array to new coordinates by interpolation.
#'
#' @description
#' Note that interpolation near boundaries differs from the scipy function,
#' because we fixed an outstanding bug
#' [scipy/issues/2640](https://github.com/scipy/scipy/issues/2640).
#'
#' @returns
#' Output input or batch of inputs.
#'
#' @param inputs
#' The input array.
#'
#' @param coordinates
#' The coordinates at which `inputs` is evaluated.
#'
#' @param order
#' The order of the spline interpolation. The order must be `0` or
#' `1`. `0` indicates the nearest neighbor and `1` indicates the linear
#' interpolation.
#'
#' @param fill_mode
#' Points outside the boundaries of the inputs are filled
#' according to the given mode. Available methods are `"constant"`,
#' `"nearest"`, `"wrap"` and `"mirror"` and `"reflect"`. Defaults to
#' `"constant"`.
#' - `"constant"`: `(k k k k | a b c d | k k k k)`
#'     inputs is extended by filling all values beyond
#'     the edge with the same constant value k specified by
#'     `fill_value`.
#' - `"nearest"`: `(a a a a | a b c d | d d d d)`
#'     inputs is extended by the nearest pixel.
#' - `"wrap"`: `(a b c d | a b c d | a b c d)`
#'     inputs is extended by wrapping around to the opposite edge.
#' - `"mirror"`: `(c d c b | a b c d | c b a b)`
#'     inputs is extended by mirroring about the edge.
#' - `"reflect"`: `(d c b a | a b c d | d c b a)`
#'     inputs is extended by reflecting about the edge of the last
#'     pixel.
#'
#' @param fill_value
#' Value used for points outside the boundaries of the inputs
#' if `fill_mode = "constant"`. Defaults to `0`.
#'
#' @export
#' @family image ops
#' @family image utils
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/image/map_coordinates>
#' @tether keras.ops.image.map_coordinates
op_image_map_coordinates <-
function (inputs, coordinates, order, fill_mode = "constant",
    fill_value = 0L)
{
    args <- capture_args(list(fill_value = as_integer))
    # "inputs" renamed from "input" in Keras 3.4.0
    args <- args_to_positional(args, "inputs")
    do.call(ops$image$map_coordinates, args)
}


#' Pad `images` with zeros to the specified `height` and `width`.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' images <- random_uniform(c(15, 25, 3))
#' padded_images <- op_image_pad(
#'     images, 2, 3, target_height = 20, target_width = 30
#' )
#' shape(padded_images)
#' ```
#'
#' ```{r}
#' batch_images <- random_uniform(c(2, 15, 25, 3))
#' padded_batch <- op_image_pad(batch_images, 2, 3,
#'                              target_height = 20,
#'                              target_width = 30)
#' shape(padded_batch)
#' ```
#'
#' @returns
#' Padded image or batch of images.
#'
#' @param images
#' Input image or batch of images. Must be 3D or 4D.
#'
#' @param top_padding
#' Number of rows of zeros to add on top.
#'
#' @param bottom_padding
#' Number of rows of zeros to add at the bottom.
#'
#' @param left_padding
#' Number of columns of zeros to add on the left.
#'
#' @param right_padding
#' Number of columns of zeros to add on the right.
#'
#' @param target_height
#' Height of output images.
#'
#' @param target_width
#' Width of output images.
#'
#' @param data_format
#' A string specifying the data format of the input tensor.
#' It can be either `"channels_last"` or `"channels_first"`.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch, height, width, channels)`, while `"channels_first"`
#' corresponds to inputs with shape `(batch, channels, height, width)`.
#' If not specified, the value will default to
#' `config_image_data_format()`.
#'
#' @export
#' @family image ops
#' @family image utils
#' @family ops
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/image/pad_images>
#'
#' @tether keras.ops.image.pad_images
op_image_pad <-
function (images, top_padding = NULL, left_padding = NULL,
          bottom_padding = NULL, right_padding = NULL,
          target_height = NULL, target_width = NULL, data_format = NULL)
{
    args <- capture_args(list(images = as_integer, top_padding = as_integer,
        bottom_padding = as_integer, left_padding = as_integer,
        right_padding = as_integer, target_height = as_integer,
        target_width = as_integer))
    do.call(ops$image$pad_images, args)
}


#'
#' Resize images to size using the specified interpolation method.
#'
#' @description
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(2, 4, 4, 3)) # batch of 2 RGB images
#' y <- op_image_resize(x, c(2, 2))
#' shape(y)
#' ```
#'
#' ```{r}
#' x <- random_uniform(c(4, 4, 3)) # single RGB image
#' y <- op_image_resize(x, c(2, 2))
#' shape(y)
#' ```
#'
#' ```{r}
#' x <- random_uniform(c(2, 3, 4, 4)) # batch of 2 RGB images
#' y <- op_image_resize(x, c(2, 2), data_format = "channels_first")
#' shape(y)
#' ```
#'
#' @returns
#' Resized image or batch of images.
#'
#' @param images
#' Input image or batch of images. Must be 3D or 4D.
#'
#' @param size
#' Size of output image in `(height, width)` format.
#'
#' @param interpolation
#' Interpolation method. Available methods are `"nearest"`,
#' `"bilinear"`, and `"bicubic"`. Defaults to `"bilinear"`.
#'
#' @param antialias
#' Whether to use an antialiasing filter when downsampling an
#' image. Defaults to `FALSE`.
#'
#' @param crop_to_aspect_ratio
#' If `TRUE`, resize the images without aspect
#' ratio distortion. When the original aspect ratio differs
#' from the target aspect ratio, the output image will be
#' cropped so as to return the
#' largest possible window in the image (of size `(height, width)`)
#' that matches the target aspect ratio. By default
#' (`crop_to_aspect_ratio=FALSE`), aspect ratio may not be preserved.
#'
#' @param pad_to_aspect_ratio
#' If `TRUE`, pad the images without aspect
#' ratio distortion. When the original aspect ratio differs
#' from the target aspect ratio, the output image will be
#' evenly padded on the short side.
#'
#' @param fill_mode
#' When using `pad_to_aspect_ratio=TRUE`, padded areas
#' are filled according to the given mode. Only `"constant"` is
#' supported at this time
#' (fill with constant value, equal to `fill_value`).
#'
#' @param fill_value
#' Float. Padding value to use when `pad_to_aspect_ratio=TRUE`.
#'
#' @param data_format
#' A string specifying the data format of the input tensor.
#' It can be either `"channels_last"` or `"channels_first"`.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch, height, width, channels)`, while `"channels_first"`
#' corresponds to inputs with shape `(batch, channels, height, width)`.
#' If not specified, the value will default to
#' `config_image_data_format()`.
#'
#'
#' @export
#' @family image ops
#' @family image utils
#' @family ops
#' @seealso
#' + <https://keras.io/api/ops/image#resize-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/image/resize>
#' @tether keras.ops.image.resize
op_image_resize <-
function (images, size, interpolation = "bilinear", antialias = FALSE,
          crop_to_aspect_ratio = FALSE, pad_to_aspect_ratio = FALSE,
          fill_mode = "constant", fill_value = 0, data_format = NULL)
{
    args <- capture_args(list(size = as_integer))
    args <- args_to_positional(args, "images")
    do.call(ops$image$resize, args)
}


#' Crop `images` to a specified `height` and `width`.
#'
#' @description
#'
#' # Examples
#' ```r
#' images <- op_reshape(op_arange(1, 28, dtype="float32"), c(3, 3, 3))
#' images[, , 1] # print the first channel of the images
#'
#' cropped_images <- op_image_crop(images, 0, 0, 2, 2)
#' cropped_images[, , 1] # print the first channel of the cropped images
#' ```
#'
#' @returns
#' Cropped image or batch of images.
#'
#' @param images
#' Input image or batch of images. Must be 3D or 4D.
#'
#' @param top_cropping
#' Number of columns to crop from the top.
#'
#' @param bottom_cropping
#' Number of columns to crop from the bottom.
#'
#' @param left_cropping
#' Number of columns to crop from the left.
#'
#' @param right_cropping
#' Number of columns to crop from the right.
#'
#' @param target_height
#' Height of the output images.
#'
#' @param target_width
#' Width of the output images.
#'
#' @param data_format
#' A string specifying the data format of the input tensor.
#' It can be either `"channels_last"` or `"channels_first"`.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch, height, width, channels)`, while `"channels_first"`
#' corresponds to inputs with shape `(batch, channels, height, width)`.
#' If not specified, the value will default to
#' `config_image_data_format()`.
#'
#' @export
#' @family image ops
#' @family image utils
#' @family ops
#' @tether keras.ops.image.crop_images
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/ops/image/crop_images>
op_image_crop <-
function (images, top_cropping = NULL, left_cropping = NULL,
          bottom_cropping = NULL, right_cropping = NULL,
          target_height = NULL, target_width = NULL,
          data_format = NULL) {
  args <- capture_args(list(
      top_cropping = as_integer,
      left_cropping = as_integer,
      bottom_cropping = as_integer,
      right_cropping = as_integer,
      target_height = as_integer,
      target_width = as_integer
  ))
  do.call(ops$image$crop_images, args)
}

#' Convert RGB images to grayscale.
#'
#' @description
#' This function converts RGB images to grayscale images. It supports both
#' 3D and 4D tensors, where the last dimension represents channels.
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(2, 4, 4, 3))
#' y <- op_image_rgb_to_grayscale(x)
#' shape(y)
#' ```
#'
#' ```{r}
#' x <- random_uniform(c(4, 4, 3)) # Single RGB image
#' y = op_image_rgb_to_grayscale(x)
#' shape(y)
#' ```
#'
#' ```{r}
#' x <- random_uniform(c(2, 3, 4, 4))
#' y <- op_image_rgb_to_grayscale(x, data_format="channels_first")
#' shape(y)
#' ```
#'
#' @returns
#' Grayscale image or batch of grayscale images.
#'
#' @param images
#' Input image or batch of images. Must be 3D or 4D.
#'
#' @param data_format
#' A string specifying the data format of the input tensor.
#' It can be either `"channels_last"` or `"channels_first"`.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch, height, width, channels)`, while `"channels_first"`
#' corresponds to inputs with shape `(batch, channels, height, width)`.
#' If not specified, the value will default to
#' `config_image_data_format()`.
#'
#' @export
#' @family image ops
#' @family image utils
#' @family ops
#' @tether keras.ops.image.rgb_to_grayscale
op_image_rgb_to_grayscale <-
function (images, data_format = NULL) {
  ops$image$rgb_to_grayscale(images, data_format)
}



#' Convert HSV images to RGB.
#'
#' @description
#' `images` must be of float dtype, and the output is only well defined if the
#' values in `images` are in `[0, 1]`.
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(2, 4, 4, 3))
#' y <- op_image_hsv_to_rgb(x)
#' shape(y)
#' ```
#'
#' ```{r}
#' x <- random_uniform(c(4, 4, 3)) # Single HSV image
#' y <- op_image_hsv_to_rgb(x)
#' shape(y)
#' ```
#'
#' ```{r}
#' x <- random_uniform(c(2, 3, 4, 4))
#' y <- op_image_hsv_to_rgb(x, data_format="channels_first")
#' shape(y)
#' ```
#'
#' @returns
#' RGB image or batch of RGB images.
#'
#' @param images
#' Input image or batch of images. Must be 3D or 4D.
#'
#' @param data_format
#' A string specifying the data format of the input tensor.
#' It can be either `"channels_last"` or `"channels_first"`.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch, height, width, channels)`, while `"channels_first"`
#' corresponds to inputs with shape `(batch, channels, height, width)`.
#' If not specified, the value will default to
#' `config_image_data_format()`.
#'
#' @family image ops
#' @family image utils
#' @family ops
#' @export
#' @tether keras.ops.image.hsv_to_rgb
op_image_hsv_to_rgb <-
function (images, data_format = NULL) {
  ops$image$hsv_to_rgb(images, data_format)
}

#' Convert RGB images to HSV.
#'
#' @description
#' `images` must be of float dtype, and the output is only well defined if the
#' values in `images` are in `[0, 1]`.
#'
#' All HSV values are in `[0, 1]`. A hue of `0` corresponds to pure red, `1/3`
#' is pure green, and `2/3` is pure blue.
#'
#' # Examples
#' ```{r}
#' x <- random_uniform(c(2, 4, 4, 3))
#' y <- op_image_rgb_to_hsv(x)
#' shape(y)
#' ```
#'
#' ```{r}
#' x <- random_uniform(c(4, 4, 3)) # Single RGB image
#' y <- op_image_rgb_to_hsv(x)
#' shape(y)
#' ```
#'
#' ```{r}
#' x <- random_uniform(c(2, 3, 4, 4))
#' y <- op_image_rgb_to_hsv(x, data_format = "channels_first")
#' shape(y)
#' ```
#'
#' @returns
#' HSV image or batch of HSV images.
#'
#' @param images
#' Input image or batch of images. Must be 3D or 4D.
#'
#' @param data_format
#' A string specifying the data format of the input tensor.
#' It can be either `"channels_last"` or `"channels_first"`.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch, height, width, channels)`, while `"channels_first"`
#' corresponds to inputs with shape `(batch, channels, height, width)`.
#' If not specified, the value will default to
#' `config_image_data_format()`.
#'
#' @family image ops
#' @family image utils
#' @family ops
#' @export
#' @tether keras.ops.image.rgb_to_hsv
op_image_rgb_to_hsv <-
function (images, data_format = NULL) {
  ops$image$rgb_to_hsv(images, data_format)
}


#' Applies a Gaussian blur to the image(s).
#'
#' @description
#'
#' # Examples
#'
#' ```{r}
#' x <- op_ones(c(2, 64, 80, 3))  # batch of 2 RGB images
#' y <- op_image_gaussian_blur(x)
#' op_shape(y)
#' ```
#'
#' ```{r}
#' x <- op_ones(c(64, 80, 3))  # single RGB image
#' y <- op_image_gaussian_blur(x)
#' op_shape(y)
#' ```
#'
#' ```{r}
#' x <- op_ones(c(2, 3, 64, 80))  # batch of 2 RGB images, channels_first
#' y <- op_image_gaussian_blur(x, data_format = "channels_first")
#' op_shape(y)
#' ```
#'
#' @returns
#' Blurred image or batch of images.
#'
#' @param images
#' Input image or batch of images. Must be 3D or 4D.
#'
#' @param kernel_size
#' A tuple of two integers, specifying the height and width
#' of the Gaussian kernel.
#'
#' @param sigma
#' A tuple of two floats, specifying the standard deviation of
#' the Gaussian kernel along height and width.
#'
#' @param data_format
#' A string specifying the data format of the input tensor.
#' It can be either `"channels_last"` or `"channels_first"`.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch, height, width, channels)`, while `"channels_first"`
#' corresponds to inputs with shape `(batch, channels, height, width)`.
#' If not specified, the value will default to
#' `keras.config.image_data_format`.
#'
#' @export
#' @tether keras.ops.image.gaussian_blur
#' @family image ops
#' @family image utils
#' @family ops
op_image_gaussian_blur <-
function (images, kernel_size = list(3L, 3L), sigma = list(1,
    1), data_format = NULL) {
  args <- capture_args(list(kernel_size = as_integer_tuple, sigma = as_tuple))
  do.call(keras$ops$image$gaussian_blur, args)
}

#' Applies a perspective transformation to the image(s).
#'
#' @description
#'
#' # Examples
#'
#' ```{r}
#' # Batch of 2 RGB images (channels_last)
#' x <- op_ones(c(2, 64, 80, 3))
#' start_points <- op_stack(list(
#'   rbind(c(0, 0), c(0, 64), c(80, 0), c(80, 64)),
#'   rbind(c(0, 0), c(0, 64), c(80, 0), c(80, 64))
#' ))
#' end_points <- op_stack(list(
#'   rbind(c(3, 5), c(7, 64), c(76, -10), c(84, 61)),
#'   rbind(c(8, 10), c(10, 61), c(65, 3), c(88, 43))
#' ))
#' y <- op_image_perspective_transform(x, start_points, end_points)
#' op_shape(y)
#' ```
#'
#' ```{r}
#' # Single RGB image (channels_last)
#' x <- op_ones(c(64, 80, 3))
#' start_points <- rbind(c(0, 0), c(0, 64), c(80, 0), c(80, 64))
#' end_points <- rbind(c(3, 5), c(7, 64), c(76, -10), c(84, 61))
#' y <- op_image_perspective_transform(x, start_points, end_points)
#' op_shape(y)
#' ```
#'
#' ```{r}
#' # Batch of 2 RGB images (channels_first)
#' x <- op_ones(c(2, 3, 64, 80))
#' start_points <- op_stack(list(
#'   rbind(c(0, 0), c(0, 64), c(80, 0), c(80, 64)),
#'   rbind(c(0, 0), c(0, 64), c(80, 0), c(80, 64))
#' ))
#' end_points <- op_stack(list(
#'   rbind(c(3, 5), c(7, 64), c(76, -10), c(84, 61)),
#'   rbind(c(8, 10), c(10, 61), c(65, 3), c(88, 43))
#' ))
#' y <- op_image_perspective_transform(
#'   x, start_points, end_points,
#'   data_format = "channels_first"
#' )
#' op_shape(y)
#' ```
#'
#' @returns
#' Applied perspective transform image or batch of images.
#'
#' @param images
#' Input image or batch of images. Must be 3D or 4D.
#'
#' @param start_points
#' A tensor of shape `(N, 4, 2)` or `(4, 2)`,
#' representing the source points in the original image
#' that define the transformation.
#'
#' @param end_points
#' A tensor of shape `(N, 4, 2)` or `(4, 2)`,
#' representing the target points in the output image
#' after transformation.
#'
#' @param interpolation
#' Interpolation method. Available methods are `"nearest"`,
#' and `"bilinear"`. Defaults to `"bilinear"`.
#'
#' @param fill_value
#' Value used for points outside the boundaries of the input if
#' extrapolation is needed. Defaults to `0`.
#'
#' @param data_format
#' A string specifying the data format of the input tensor.
#' It can be either `"channels_last"` or `"channels_first"`.
#' `"channels_last"` corresponds to inputs with shape
#' `(batch, height, width, channels)`, while `"channels_first"`
#' corresponds to inputs with shape `(batch, channels, height, width)`.
#' If not specified, the value will default to
#' `keras.config.image_data_format`.
#'
#' @export
#' @tether keras.ops.image.perspective_transform
#' @family image ops
#' @family image utils
#' @family ops
op_image_perspective_transform <-
function (images, start_points, end_points, interpolation = "bilinear",
    fill_value = 0L, data_format = NULL)
{
    args <- capture_args(list(fill_value = as_integer))
    do.call(keras$ops$image$perspective_transform, args)
}
