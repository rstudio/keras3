Turns positive integers (indexes) into dense vectors of fixed size.

@description
e.g. `rbind(4L, 20L)` \eqn{\rightarrow}{->} `rbind(c(0.25, 0.1), c(0.6, -0.2))`

This layer can only be used on positive integer inputs of a fixed range.

# Examples

```r
model <- keras_model_sequential() |>
  layer_embedding(1000, 64)

# The model will take as input an integer matrix of size (batch,input_length),
# and the largest integer (i.e. word index) in the input
# should be no larger than 999 (vocabulary size).
# Now model$output_shape is (NA, 10, 64), where `NA` is the batch
# dimension.

input_array <- random_integer(shape = c(32, 10), minval = 0, maxval = 1000)
model |> compile('rmsprop', 'mse')
output_array <- model |> predict(input_array, verbose = 0)
dim(output_array)    # (32, 10, 64)
```

```
## [1] 32 10 64
```

# Input Shape
2D tensor with shape: `(batch_size, input_length)`.

# Output Shape
3D tensor with shape: `(batch_size, input_length, output_dim)`.

@param input_dim
Integer. Size of the vocabulary,
i.e. maximum integer index + 1.

@param output_dim
Integer. Dimension of the dense embedding.

@param embeddings_initializer
Initializer for the `embeddings`
matrix (see `keras3::initializer_*`).

@param embeddings_regularizer
Regularizer function applied to
the `embeddings` matrix (see `keras3::regularizer_*`).

@param embeddings_constraint
Constraint function applied to
the `embeddings` matrix (see `keras3::constraint_*`).

@param mask_zero
Boolean, whether or not the input value 0 is a special
"padding" value that should be masked out.
This is useful when using recurrent layers which
may take variable length input. If this is `TRUE`,
then all subsequent layers in the model need
to support masking or an exception will be raised.
If `mask_zero` is set to `TRUE`, as a consequence,
index 0 cannot be used in the vocabulary (`input_dim` should
equal size of vocabulary + 1).

@param object
Object to compose the layer with. A tensor, array, or sequential model.

@param ...
For forward/backward compatability.

@export
@family core layers
@family layers
@seealso
+ <https:/keras.io/api/layers/core_layers/embedding#embedding-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding>
