Applies the given transform(s) to the image(s).

@description

# Examples
```python
x = np.random.random((2, 64, 80, 3)) # batch of 2 RGB images
transform = np.array(
    [
        [1.5, 0, -20, 0, 1.5, -16, 0, 0],  # zoom
        [1, 0, -20, 0, 1, -16, 0, 0],  # translation
    ]
)
y = keras.ops.image.affine_transform(x, transform)
y.shape
# (2, 64, 80, 3)
```

```python
x = np.random.random((64, 80, 3)) # single RGB image
transform = np.array([1.0, 0.5, -20, 0.5, 1.0, -16, 0, 0])  # shear
y = keras.ops.image.affine_transform(x, transform)
y.shape
# (64, 80, 3)
```

```python
x = np.random.random((2, 3, 64, 80)) # batch of 2 RGB images
transform = np.array(
    [
        [1.5, 0, -20, 0, 1.5, -16, 0, 0],  # zoom
        [1, 0, -20, 0, 1, -16, 0, 0],  # translation
    ]
)
y = keras.ops.image.affine_transform(x, transform,
    data_format="channels_first")
y.shape
# (2, 3, 64, 80)
```

@returns
Applied affine transform image or batch of images.

@param image
Input image or batch of images. Must be 3D or 4D.

@param transform
Projective transform matrix/matrices. A vector of length 8 or
tensor of size N x 8. If one row of transform is
`[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps the output point
`(x, y)` to a transformed input point
`(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
where `k = c0 x + c1 y + 1`. The transform is inverted compared to
the transform mapping input points to output points. Note that
gradients are not backpropagated into transformation parameters.
Note that `c0` and `c1` are only effective when using TensorFlow
backend and will be considered as `0` when using other backends.

@param interpolation
Interpolation method. Available methods are `"nearest"`,
and `"bilinear"`. Defaults to `"bilinear"`.

@param fill_mode
Points outside the boundaries of the input are filled
according to the given mode. Available methods are `"constant"`,
`"nearest"`, `"wrap"` and `"reflect"`. Defaults to `"constant"`.
- `"reflect"`: `(d c b a | a b c d | d c b a)`
    The input is extended by reflecting about the edge of the last
    pixel.
- `"constant"`: `(k k k k | a b c d | k k k k)`
    The input is extended by filling all values beyond
    the edge with the same constant value k specified by
    `fill_value`.
- `"wrap"`: `(a b c d | a b c d | a b c d)`
    The input is extended by wrapping around to the opposite edge.
- `"nearest"`: `(a a a a | a b c d | d d d d)`
    The input is extended by the nearest pixel.

@param fill_value
Value used for points outside the boundaries of the input if
`fill_mode="constant"`. Defaults to `0`.

@param data_format
string, either `"channels_last"` or `"channels_first"`.
The ordering of the dimensions in the inputs. `"channels_last"`
corresponds to inputs with shape `(batch, height, width, channels)`
while `"channels_first"` corresponds to inputs with shape
`(batch, channels, height, weight)`. It defaults to the
`image_data_format` value found in your Keras config file at
`~/.keras/keras.json`. If you never set it, then it will be
`"channels_last"`.

@export
@family image ops
@family image utils
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/image#affinetransform-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/image/affine_transform>
