keras.ops.while_loop
__signature__
(
  cond,
  body,
  loop_vars,
  maximum_iterations=None
)
__doc__
While loop implementation.

Args:
    cond: A callable that represents the termination condition of the loop.
        Must have the same number of args as `loop_vars`, and return a bool.
    body: A callable that represents the loop body. Must have the same
        number of args as `loop_vars`, and return a list/tuple of the same
        length, shape and dtype as `loop_vars`.
    loop_vars: A list/tuple of tensors, the loop variables.
    maximum_iterations: Optional maximum number of iterations of the while
        loop to run. If provided, the `cond` output is AND-ed with an
        additional condition ensuring the number of iterations executed is
        no greater than `maximum_iterations`.

Returns:
    A list/tuple of tensors, has the same shape and dtype as `inputs`.

Examples:

>>> i = 0
>>> cond = lambda i: i < 10
>>> body = lambda i: i + 1
>>> keras.ops.while_loop(cond, body, [i])[0]
10
