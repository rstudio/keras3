Shuffle the elements of a tensor uniformly at random along an axis.

@param x
The tensor to be shuffled.

@param axis
An integer specifying the axis along which to shuffle. Defaults to
`0`.

@param seed
A Python integer or instance of
`keras.random.SeedGenerator`.
Used to make the behavior of the initializer
deterministic. Note that an initializer seeded with an integer
or None (unseeded) will produce the same random values
across multiple calls. To get different random values
across multiple calls, use as seed an instance
of `keras.random.SeedGenerator`.

@export
@family random
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/random/shuffle>
