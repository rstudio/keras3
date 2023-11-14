Concatenates a list of inputs.

@description
It takes as input a list of tensors, all of the same shape except
for the concatenation axis, and returns a single tensor that is the
concatenation of all inputs.

# Examples

```r
x <- k_arange(20) |> k_reshape(c(2, 2, 5))
y <- k_arange(20, 40) |> k_reshape(c(2, 2, 5))
layer_concatenate(x, y, axis = 2)
```

```
## tf.Tensor(
## [[[ 0.  1.  2.  3.  4.]
##   [ 5.  6.  7.  8.  9.]
##   [20. 21. 22. 23. 24.]
##   [25. 26. 27. 28. 29.]]
##
##  [[10. 11. 12. 13. 14.]
##   [15. 16. 17. 18. 19.]
##   [30. 31. 32. 33. 34.]
##   [35. 36. 37. 38. 39.]]], shape=(2, 4, 5), dtype=float32)
```
Usage in a Keras model:


```r
x1 <- k_arange(10)     |> k_reshape(c(5, 2)) |> layer_dense(8)
x2 <- k_arange(10, 20) |> k_reshape(c(5, 2)) |> layer_dense(8)
y <- layer_concatenate(x1, x2)
```

@returns
    A tensor, the concatenation of the inputs alongside axis `axis`.

@param axis
Axis along which to concatenate.

@param ...
Standard layer keyword arguments.

@param inputs
layers to combine

@export
@family merging layers
@seealso
+ <https:/keras.io/api/layers/merging_layers/concatenate#concatenate-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate>
