Normalizes an array.

@description
If the input is a NumPy array, a NumPy array will be returned.
If it's a backend tensor, a backend tensor will be returned.

# Returns
    A normalized copy of the array.

@param x Array to normalize.
@param axis axis along which to normalize.
@param order Normalization order (e.g. `order=2` for L2 norm).

@export
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/normalize>
