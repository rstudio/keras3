A preprocessing layer which randomly translates images during training.

@description
This layer will apply random translations to each image during training,
filling empty space according to `fill_mode`.

Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
of integer or floating point dtype. By default, the layer will output
floats.

# Input Shape
3D (unbatched) or 4D (batched) tensor with shape:
`(..., height, width, channels)`, in `"channels_last"` format,
or `(..., channels, height, width)`, in `"channels_first"` format.

# Output Shape
3D (unbatched) or 4D (batched) tensor with shape:
    `(..., target_height, target_width, channels)`,
    or `(..., channels, target_height, target_width)`,
    in `"channels_first"` format.

**Note:** This layer is safe to use inside a `tf.data` pipeline
(independently of which backend you're using).

@param height_factor
a float represented as fraction of value, or a tuple of
size 2 representing lower and upper bound for shifting vertically. A
negative value means shifting image up, while a positive value means
shifting image down. When represented as a single positive float,
this value is used for both the upper and lower bound. For instance,
`height_factor=(-0.2, 0.3)` results in an output shifted by a random
amount in the range `[-20%, +30%]`. `height_factor=0.2` results in
an output height shifted by a random amount in the range
`[-20%, +20%]`.

@param width_factor
a float represented as fraction of value, or a tuple of
size 2 representing lower and upper bound for shifting horizontally.
A negative value means shifting image left, while a positive value
means shifting image right. When represented as a single positive
float, this value is used for both the upper and lower bound. For
instance, `width_factor=(-0.2, 0.3)` results in an output shifted
left by 20%, and shifted right by 30%. `width_factor=0.2` results
in an output height shifted left or right by 20%.

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
Note that when using torch backend, `"reflect"` is redirected to
`"mirror"` `(c d c b | a b c d | c b a b)` because torch does not
support `"reflect"`.
Note that torch backend does not support `"wrap"`.

@param interpolation
Interpolation mode. Supported values: `"nearest"`,
`"bilinear"`.

@param seed
Integer. Used to create a random seed.

@param fill_value
a float represents the value to be filled outside the
boundaries when `fill_mode="constant"`.

@param data_format
string, either `"channels_last"` or `"channels_first"`.
The ordering of the dimensions in the inputs. `"channels_last"`
corresponds to inputs with shape `(batch, height, width, channels)`
while `"channels_first"` corresponds to inputs with shape
`(batch, channels, height, width)`. It defaults to the
`image_data_format` value found in your Keras config file at
`~/.keras/keras.json`. If you never set it, then it will be
`"channels_last"`.

@param ...
Base layer keyword arguments, such as `name` and `dtype`.

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@export
@family preprocessing layers
@family layers
@seealso
+ <https:/keras.io/api/layers/preprocessing_layers/image_augmentation/random_translation#randomtranslation-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomTranslation>
