For loop implementation.

@description

# Examples

```r
lower <- 0L
upper <- 10L
body_fun <- function(i, state) state + i
init_state <- 0L
final_state <- k_fori_loop(lower, upper, body_fun, init_state)
final_state
```

```
## tf.Tensor(45, shape=(), dtype=int32)
```

@returns
The final state after the loop.

@param lower
The initial value of the loop variable.

@param upper
The upper bound of the loop variable.

@param body_fun
A callable that represents the loop body. Must take two
arguments: the loop variable and the loop state. The loop state
should be updated and returned by this function.

@param init_val
The initial value of the loop state.

@export
@family core ops
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/fori_loop>
