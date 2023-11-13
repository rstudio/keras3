A preprocessing layer that normalizes continuous features.

@description
This layer will shift and scale inputs into a distribution centered around
0 with standard deviation 1. It accomplishes this by precomputing the mean
and variance of the data, and calling `(input - mean) / sqrt(var)` at
runtime.

The mean and variance values for the layer must be either supplied on
construction or learned via `adapt()`. `adapt()` will compute the mean and
variance of the data and store them as the layer's weights. `adapt()` should
be called before `fit()`, `evaluate()`, or `predict()`.

# Examples
Calculate a global mean and variance by analyzing the dataset in `adapt()`.


```r
adapt_data <- k_array(c(1., 2., 3., 4., 5.), dtype='float32')
input_data <- k_array(c(1., 2., 3.), dtype='float32')
layer <- layer_normalization(axis = NULL)
layer %>% adapt(adapt_data)
layer(input_data)
```

```
## tf.Tensor([-1.4142135  -0.70710677  0.        ], shape=(3), dtype=float32)
```

Calculate a mean and variance for each index on the last axis.


```r
adapt_data <- k_array(rbind(c(0., 7., 4.),
                       c(2., 9., 6.),
                       c(0., 7., 4.),
                       c(2., 9., 6.)), dtype='float32')
input_data <- k_array(matrix(c(0., 7., 4.), nrow = 1), dtype='float32')
layer <- layer_normalization(axis=-1)
layer %>% adapt(adapt_data)
layer(input_data)
```

```
## tf.Tensor([[-1. -1. -1.]], shape=(1, 3), dtype=float32)
```

Pass the mean and variance directly.


```r
input_data <- k_array(rbind(1, 2, 3), dtype='float32')
layer <- layer_normalization(mean=3., variance=2.)
layer(input_data)
```

```
## tf.Tensor(
## [[-1.4142135 ]
##  [-0.70710677]
##  [ 0.        ]], shape=(3, 1), dtype=float32)
```

Use the layer to de-normalize inputs (after adapting the layer).


```r
adapt_data <- k_array(rbind(c(0., 7., 4.),
                       c(2., 9., 6.),
                       c(0., 7., 4.),
                       c(2., 9., 6.)), dtype='float32')
input_data <- k_array(c(1., 2., 3.), dtype='float32')
layer <- layer_normalization(axis=-1, invert=TRUE)
layer %>% adapt(adapt_data)
layer(input_data)
```

```
## tf.Tensor([[ 2. 10.  8.]], shape=(1, 3), dtype=float32)
```

@param axis Integer, list of integers, or NULL. The axis or axes that should
    have a separate mean and variance for each index in the shape.
    For example, if shape is `(NULL, 5)` and `axis=1`, the layer will
    track 5 separate mean and variance values for the last axis.
    If `axis` is set to `NULL`, the layer will normalize
    all elements in the input by a scalar mean and variance.
    When `-1`, the last axis of the input is assumed to be a
    feature dimension and is normalized per index.
    Note that in the specific case of batched scalar inputs where
    the only axis is the batch axis, the default will normalize
    each index in the batch separately.
    In this case, consider passing `axis=NULL`. Defaults to `-1`.
@param mean The mean value(s) to use during normalization. The passed value(s)
    will be broadcast to the shape of the kept axes above;
    if the value(s) cannot be broadcast, an error will be raised when
    this layer's `build()` method is called.
@param variance The variance value(s) to use during normalization. The passed
    value(s) will be broadcast to the shape of the kept axes above;
    if the value(s) cannot be broadcast, an error will be raised when
    this layer's `build()` method is called.
@param invert If `TRUE`, this layer will apply the inverse transformation
    to its inputs: it would turn a normalized input back into its
    original form.
@param object Object to compose the layer with. A tensor, array, or sequential model.
@param ... Passed on to the Python callable

@export
@family preprocessing layers
@seealso
+ <https:/keras.io/api/layers/preprocessing_layers/numerical/normalization#normalization-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization>

