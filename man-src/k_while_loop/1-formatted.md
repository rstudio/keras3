While loop implementation.

@description

# Examples
```python
i = 0
cond = lambda i: i < 10
body = lambda i: i + 1
keras.ops.while_loop(cond, body, [i])[0]
# 10
```

@returns
A list/tuple of tensors, has the same shape and dtype as `inputs`.

@param cond
A callable that represents the termination condition of the loop.
Must have the same number of args as `loop_vars`, and return a bool.

@param body
A callable that represents the loop body. Must have the same
number of args as `loop_vars`, and return a list/tuple of the same
length, shape and dtype as `loop_vars`.

@param loop_vars
A list/tuple of tensors, the loop variables.

@param maximum_iterations
Optional maximum number of iterations of the while
loop to run. If provided, the `cond` output is AND-ed with an
additional condition ensuring the number of iterations executed is
no greater than `maximum_iterations`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/core#whileloop-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/while_loop>
