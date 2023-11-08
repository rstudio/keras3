keras.random.shuffle
__signature__
(x, axis=0, seed=None)
__doc__
Shuffle the elements of a tensor uniformly at random along an axis.

Args:
    x: The tensor to be shuffled.
    axis: An integer specifying the axis along which to shuffle. Defaults to
        `0`.
    seed: A Python integer or instance of
        `keras.random.SeedGenerator`.
        Used to make the behavior of the initializer
        deterministic. Note that an initializer seeded with an integer
        or None (unseeded) will produce the same random values
        across multiple calls. To get different random values
        across multiple calls, use as seed an instance
        of `keras.random.SeedGenerator`.
