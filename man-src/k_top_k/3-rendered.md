Finds the top-k values and their indices in a tensor.

@description

# Examples

```r
x <- k_array(c(5, 2, 7, 1, 9, 3), "int32")
k_top_k(x, k = 3)
```

```
## $values
## tf.Tensor([9 7 5], shape=(3), dtype=int32)
##
## $indices
## tf.Tensor([4 2 0], shape=(3), dtype=int32)
```


```r
c(values, indices) %<-% k_top_k(x, k = 3)
values
```

```
## tf.Tensor([9 7 5], shape=(3), dtype=int32)
```

```r
indices
```

```
## tf.Tensor([4 2 0], shape=(3), dtype=int32)
```

@returns
A list containing two tensors. The first tensor contains the
top-k values, and the second tensor contains the indices of the
top-k values in the input tensor.

@param x
Input tensor.

@param k
An integer representing the number of top elements to retrieve.

@param sorted
A boolean indicating whether to sort the output in
descending order. Defaults to`TRUE`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/core#topk-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/top_k>
