Append tensor `x2` to the end of tensor `x1`.

@description

# Examples

```r
x1 <- k_convert_to_tensor(c(1, 2, 3))
x2 <- k_convert_to_tensor(rbind(c(4, 5, 6), c(7, 8, 9)))
k_append(x1, x2)
```

```
## tf.Tensor([1. 2. 3. 4. 5. 6. 7. 8. 9.], shape=(9), dtype=float64)
```

When `axis` is specified, `x1` and `x2` must have compatible shapes.

```r
x1 <- k_convert_to_tensor(rbind(c(1, 2, 3), c(4, 5, 6)))
x2 <- k_convert_to_tensor(rbind(c(7, 8, 9)))
k_append(x1, x2, axis = 1)
```

```
## tf.Tensor(
## [[1. 2. 3.]
##  [4. 5. 6.]
##  [7. 8. 9.]], shape=(3, 3), dtype=float64)
```

```r
x3 <- k_convert_to_tensor(c(7, 8, 9))
try(k_append(x1, x3, axis = 1))
```

```
## Error in py_call_impl(callable, call_args$unnamed, call_args$named) : 
##   tensorflow.python.framework.errors_impl.InvalidArgumentError: {{function_node __wrapped__ConcatV2_N_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} ConcatOp : Ranks of all input tensors should match: shape[0] = [2,3] vs. shape[1] = [3] [Op:ConcatV2] name: concat
## Run `reticulate::py_last_error()` for details.
```

@returns
A tensor with the values of `x2` appended to `x1`.

@param x1 First input tensor.
@param x2 Second input tensor.
@param axis Axis along which tensor `x2` is appended to tensor `x1`.
    If `NULL`, both tensors are flattened before use.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/append>
