Averages a list of inputs element-wise..

@description
It takes as input a list of tensors, all of the same shape,
and returns a single tensor (also of the same shape).

# Examples

```r
input_shape <- c(1, 2, 3)
x1 <- k_ones(input_shape)
x2 <- k_zeros(input_shape)
layer_average(x1, x2)
```

```
## tf.Tensor(
## [[[0.5 0.5 0.5]
##   [0.5 0.5 0.5]]], shape=(1, 2, 3), dtype=float32)
```

Usage in a Keras model:


```r
input1 <- layer_input(shape = c(16))
x1 <- input1 |> layer_dense(8, activation = 'relu')

input2 <- layer_input(shape = c(32))
x2 <- input2 |> layer_dense(8, activation = 'relu')

added <- layer_average(x1, x2)
output <- added |> layer_dense(4)

model <- keras_model(inputs = c(input1, input2), outputs = output)
```

@param ...
Passed on to the Python callable

@param inputs
layers to combine

@export
@family average merging layers
@family merging layers
@family layers
@seealso
+ <https:/keras.io/api/layers/merging_layers/average#average-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Average>
