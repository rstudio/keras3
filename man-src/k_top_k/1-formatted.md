Finds the top-k values and their indices in a tensor.

@description

# Returns
A tuple containing two tensors. The first tensor contains the
top-k values, and the second tensor contains the indices of the
top-k values in the input tensor.

# Examples
```python
x = keras.ops.convert_to_tensor([5, 2, 7, 1, 9, 3])
values, indices = top_k(x, k=3)
print(values)
# array([9 7 5], shape=(3,), dtype=int32)
print(indices)
# array([4 2 0], shape=(3,), dtype=int32)
```

@param x Input tensor.
@param k An integer representing the number of top elements to retrieve.
@param sorted A boolean indicating whether to sort the output in
descending order. Defaults to`True`.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/top_k>
