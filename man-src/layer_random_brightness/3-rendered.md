A preprocessing layer which randomly adjusts brightness during training.

@description
This layer will randomly increase/reduce the brightness for the input RGB
images. At inference time, the output will be identical to the input.
Call the layer with `training=TRUE` to adjust the brightness of the input.

**Note:** This layer is safe to use inside a `tf.data` pipeline
(independently of which backend you're using).

# Inputs
3D (HWC) or 4D (NHWC) tensor, with float or int dtype. Input pixel
values can be of any range (e.g. `[0., 1.)` or `[0, 255]`)

# Output
3D (HWC) or 4D (NHWC) tensor with brightness adjusted based on the
    `factor`. By default, the layer will output floats.
    The output value will be clipped to the range `[0, 255]`,
    the valid range of RGB colors, and
    rescaled based on the `value_range` if needed.

Sample usage:


```r
random_bright <- layer_random_brightness(factor=0.2, seed = 1)

# An image with shape [2, 2, 3]
image <- array(1:12, dim=c(2, 2, 3))

# Assume we randomly select the factor to be 0.1, then it will apply
# 0.1 * 255 to all the channel
output <- random_bright(image, training=TRUE)
output
```

```
## tf.Tensor(
## [[[39.605797 43.605797 47.605797]
##   [41.605797 45.605797 49.605797]]
##
##  [[40.605797 44.605797 48.605797]
##   [42.605797 46.605797 50.605797]]], shape=(2, 2, 3), dtype=float32)
```

@param factor
Float or a list of 2 floats between -1.0 and 1.0. The
factor is used to determine the lower bound and upper bound of the
brightness adjustment. A float value will be chosen randomly between
the limits. When -1.0 is chosen, the output image will be black, and
when 1.0 is chosen, the image will be fully white.
When only one float is provided, eg, 0.2,
then -0.2 will be used for lower bound and 0.2
will be used for upper bound.

@param value_range
Optional list of 2 floats
for the lower and upper limit
of the values of the input data.
To make no change, use `c(0.0, 1.0)`, e.g., if the image input
has been scaled before this layer. Defaults to `c(0.0, 255.0)`.
The brightness adjustment will be scaled to this range, and the
output values will be clipped to this range.

@param seed
optional integer, for fixed RNG behavior.

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
Passed on to the Python callable

@export
@family brightnes random preprocessing layers
@family random preprocessing layers
@family preprocessing layers
@family layers
@seealso
+ <https:/keras.io/api/layers/preprocessing_layers/image_augmentation/random_brightness#randombrightness-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomBrightness>

