Permutes the dimensions of the input according to a given pattern.

@description
Useful e.g. connecting RNNs and convnets.

# Input Shape
Arbitrary.

# Output Shape
Same as the input shape, but with the dimensions re-ordered according
to the specified pattern.

# Examples

```r
x <- layer_input(shape=c(10, 64))
y <- layer_permute(x, c(2, 1))
y$shape
```

```
## [[1]]
## NULL
##
## [[2]]
## [1] 64
##
## [[3]]
## [1] 10
```

@param dims
List of integers. Permutation pattern does not include the
batch dimension. Indexing starts at 1.
For instance, `c(2, 1)` permutes the first and second dimensions
of the input.

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
Passed on to the Python callable

@export
@family reshaping layers
@family layers
@seealso
+ <https:/keras.io/api/layers/reshaping_layers/permute#permute-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Permute>

