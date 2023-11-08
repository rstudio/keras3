keras.layers.Permute
__signature__
(dims, **kwargs)
__doc__
Permutes the dimensions of the input according to a given pattern.

Useful e.g. connecting RNNs and convnets.

Args:
    dims: Tuple of integers. Permutation pattern does not include the
        batch dimension. Indexing starts at 1.
        For instance, `(2, 1)` permutes the first and second dimensions
        of the input.

Input shape:
    Arbitrary.

Output shape:
    Same as the input shape, but with the dimensions re-ordered according
    to the specified pattern.

Example:

>>> x = keras.Input(shape=(10, 64))
>>> y = keras.layers.Permute((2, 1))(x)
>>> y.shape
(None, 64, 10)
