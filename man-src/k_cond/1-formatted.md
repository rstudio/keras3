Conditionally applies `true_fn` or `false_fn`.

# Returns
    The output of either `true_fn` or `false_fn` depending on pred.

@param pred Boolean scalar type
@param true_fn Callable returning the output for the `pred == True` case.
@param false_fn Callable returning the output for the `pred == False` case.

@export
@family ops
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/cond>
