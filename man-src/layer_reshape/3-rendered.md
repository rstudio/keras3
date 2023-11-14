Layer that reshapes inputs into the given shape.

@description

# Input Shape
Arbitrary, although all dimensions in the input shape must be
known/fixed. Use the keyword argument `input_shape` (list of integers,
does not include the samples/batch size axis) when using this layer as
the first layer in a model.

# Output Shape
`(batch_size, *target_shape)`

# Examples

```r
x <- layer_input(shape = 12)
y <- layer_reshape(x, c(3, 4))
y$shape
```

```
## [[1]]
## NULL
##
## [[2]]
## [1] 3
##
## [[3]]
## [1] 4
```


```r
# also supports shape inference using `-1` as dimension
y <- layer_reshape(x, c(-1, 2, 2))
y$shape
```

```
## [[1]]
## NULL
##
## [[2]]
## [1] 3
##
## [[3]]
## [1] 2
##
## [[4]]
## [1] 2
```

@param target_shape
Target shape. List of integers, does not include the
samples dimension (batch size).

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
Passed on to the Python callable

@export
@family reshaping layers
@seealso
+ <https:/keras.io/api/layers/reshaping_layers/reshape#reshape-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape>

