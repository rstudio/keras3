Return elements chosen from `x1` or `x2` depending on `condition`.

@returns
A tensor with elements from `x1` where `condition` is `True`, and
elements from `x2` where `condition` is `False`.

@param condition Where `True`, yield `x1`, otherwise yield `x2`.
@param x1 Values from which to choose when `condition` is `True`.
@param x2 Values from which to choose when `condition` is `False`.

@export
@family ops
@seealso
+ <https:/keras.io/keras_core/api/ops/numpy#where-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/ops/where>
