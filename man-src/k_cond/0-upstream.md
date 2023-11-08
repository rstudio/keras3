keras.ops.cond
__signature__
(pred, true_fn, false_fn)
__doc__
Conditionally applies `true_fn` or `false_fn`.

Args:
    pred: Boolean scalar type
    true_fn: Callable returning the output for the `pred == True` case.
    false_fn: Callable returning the output for the `pred == False` case.

Returns:
    The output of either `true_fn` or `false_fn` depending on pred.
