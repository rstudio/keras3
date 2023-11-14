A preprocessing layer which encodes integer features.

@description
This layer provides options for condensing data into a categorical encoding
when the total number of tokens are known in advance. It accepts integer
values as inputs, and it outputs a dense or sparse representation of those
inputs. For integer inputs where the total number of tokens is not known,
use `layer_integer_lookup()` instead.

**Note:** This layer is safe to use inside a `tf.data` pipeline
(independently of which backend you're using).

# Examples
**One-hot encoding data**


```r
layer <- layer_category_encoding(num_tokens = 4, output_mode = "one_hot")
x <- k_array(c(3, 2, 0, 1), "int32")
layer(x)
```

```
## tf.Tensor(
## [[0. 0. 0. 1.]
##  [0. 0. 1. 0.]
##  [1. 0. 0. 0.]
##  [0. 1. 0. 0.]], shape=(4, 4), dtype=float32)
```

**Multi-hot encoding data**


```r
layer <- layer_category_encoding(num_tokens = 4, output_mode = "multi_hot")
x <- k_array(rbind(c(0, 1),
                   c(0, 0),
                   c(1, 2),
                   c(3, 1)), "int32")
layer(x)
```

```
## tf.Tensor(
## [[1. 1. 0. 0.]
##  [1. 0. 0. 0.]
##  [0. 1. 1. 0.]
##  [0. 1. 0. 1.]], shape=(4, 4), dtype=float32)
```

**Using weighted inputs in `"count"` mode**


```r
layer <- layer_category_encoding(num_tokens = 4, output_mode = "count")
count_weights <- k_array(rbind(c(.1, .2),
                               c(.1, .1),
                               c(.2, .3),
                               c(.4, .2)))
x <- k_array(rbind(c(0, 1),
                   c(0, 0),
                   c(1, 2),
                   c(3, 1)), "int32")
layer(x, count_weights = count_weights)
#   array([[01, 02, 0. , 0. ],
#          [02, 0. , 0. , 0. ],
#          [0. , 02, 03, 0. ],
#          [0. , 02, 0. , 04]]>
```

# Call Arguments
- `inputs`: A 1D or 2D tensor of integer inputs.
- `count_weights`: A tensor in the same shape as `inputs` indicating the
    weight for each sample value when summing up in `count` mode.
    Not used in `"multi_hot"` or `"one_hot"` modes.

@param num_tokens
The total number of tokens the layer should support. All
inputs to the layer must integers in the range `0 <= value <
num_tokens`, or an error will be thrown.

@param output_mode
Specification for the output of the layer.
Values can be `"one_hot"`, `"multi_hot"` or `"count"`,
configuring the layer as follows:
    - `"one_hot"`: Encodes each individual element in the input
        into an array of `num_tokens` size, containing a 1 at the
        element index. If the last dimension is size 1, will encode
        on that dimension. If the last dimension is not size 1,
        will append a new dimension for the encoded output.
    - `"multi_hot"`: Encodes each sample in the input into a single
        array of `num_tokens` size, containing a 1 for each
        vocabulary term present in the sample. Treats the last
        dimension as the sample dimension, if input shape is
        `(..., sample_length)`, output shape will be
        `(..., num_tokens)`.
    - `"count"`: Like `"multi_hot"`, but the int array contains a
        count of the number of times the token at that index
        appeared in the sample.
For all output modes, currently only output up to rank 2 is
supported.
Defaults to `"multi_hot"`.

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
Passed on to the Python callable

@export
@family preprocessing layers
@seealso
+ <https:/keras.io/api/layers/preprocessing_layers/categorical/category_encoding#categoryencoding-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/CategoryEncoding>
