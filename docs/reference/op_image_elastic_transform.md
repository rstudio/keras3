# Applies elastic deformation to the image(s).

Apply random elastic deformation to 3D or 4D image tensors.

## Usage

``` r
op_image_elastic_transform(
  images,
  alpha = 20,
  sigma = 5,
  interpolation = "bilinear",
  fill_mode = "reflect",
  fill_value = 0,
  seed = NULL,
  data_format = NULL
)
```

## Arguments

- images:

  Input image or batch of images. Must be 3D or 4D.

- alpha:

  Scaling factor that controls the intensity of the deformation.

- sigma:

  Standard deviation of the Gaussian filter used for smoothing the
  displacement fields.

- interpolation:

  Interpolation method. Available methods are `"nearest"`, and
  `"bilinear"`. Defaults to `"bilinear"`.

- fill_mode:

  Points outside the boundaries of the input are filled according to the
  given mode. Available methods are `"constant"`, `"nearest"`, `"wrap"`
  and `"reflect"`. Defaults to `"reflect"`.

  - `"reflect"`: `(d c b a | a b c d | d c b a)` The input is extended
    by reflecting about the edge of the last pixel.

  - `"constant"`: `(k k k k | a b c d | k k k k)` The input is extended
    by filling all values beyond the edge with the same constant value
    `k` specified by `fill_value`.

  - `"wrap"`: `(a b c d | a b c d | a b c d)` The input is extended by
    wrapping around to the opposite edge.

  - `"nearest"`: `(a a a a | a b c d | d d d d)` The input is extended
    by the nearest pixel.

- fill_value:

  Value used for points outside the boundaries of the input if
  `fill_mode="constant"`. Defaults to `0`.

- seed:

  Optional integer seed for the random number generator.

- data_format:

  A string specifying the data format of the input tensor. It can be
  either `"channels_last"` or `"channels_first"`. `"channels_last"`
  corresponds to inputs with shape `(batch, height, width, channels)`,
  while `"channels_first"` corresponds to inputs with shape
  `(batch, channels, height, width)`. If not specified, the value will
  default to `keras.config.image_data_format`.

## Value

Transformed image or batch of images with elastic deformation.

## Examples

    x <- random_uniform(c(2, 64, 80, 3))  # batch of 2 RGB images
    y <- op_image_elastic_transform(x)
    op_shape(y)

    ## shape(2, 64, 80, 3)

    x <- random_uniform(c(64, 80, 3))  # single RGB image
    y <- op_image_elastic_transform(x)
    op_shape(y)

    ## shape(64, 80, 3)

    x <- random_uniform(c(2, 3, 64, 80))  # batch of 2 RGB images
    y <- op_image_elastic_transform(
      x,
      data_format = "channels_first",
      seed = 123
    )
    op_shape(y)

    ## shape(2, 3, 64, 80)
