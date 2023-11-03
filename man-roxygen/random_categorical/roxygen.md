Draws samples from a categorical distribution.

@description
This function takes as input `logits`, a 2-D input tensor with shape
(batch_size, num_classes). Each row of the input represents a categorical
distribution, with each column index containing the log-probability for a
given class.

The function will output a 2-D tensor with shape (batch_size, num_samples),
where each row contains samples from the corresponding row in `logits`.
Each column index contains an independent samples drawn from the input
distribution.

# Returns
    A 2-D tensor with (batch_size, num_samples).

@param logits 2-D Tensor with shape (batch_size, num_classes). Each row
    should define a categorical distibution with the unnormalized
    log-probabilities for all classes.
@param num_samples Int, the number of independent samples to draw for each
    row of the input. This will be the second dimension of the output
    tensor's shape.
@param dtype Optional dtype of the output tensor.
@param seed A Python integer or instance of
    `keras.random.SeedGenerator`.
    Used to make the behavior of the initializer
    deterministic. Note that an initializer seeded with an integer
    or None (unseeded) will produce the same random values
    across multiple calls. To get different random values
    across multiple calls, use as seed an instance
    of `keras.random.SeedGenerator`.

@export
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/random/categorical>
