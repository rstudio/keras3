A preprocessing layer which buckets continuous features by ranges.

This layer will place each element of its input data into one of several
contiguous ranges and output an integer index indicating which range each
element was placed in.

**Note:** This layer is safe to use inside a `tf.data` pipeline
(independently of which backend you're using).

Input shape:
    Any array of dimension 2 or higher.

Output shape:
    Same as input shape.

Arguments:
    bin_boundaries: A list of bin boundaries.
        The leftmost and rightmost bins
        will always extend to `-inf` and `inf`,
        so `bin_boundaries=[0., 1., 2.]`
        generates bins `(-inf, 0.)`, `[0., 1.)`, `[1., 2.)`,
        and `[2., +inf)`.
        If this option is set, `adapt()` should not be called.
    num_bins: The integer number of bins to compute.
        If this option is set,
        `adapt()` should be called to learn the bin boundaries.
    epsilon: Error tolerance, typically a small fraction
        close to zero (e.g. 0.01). Higher values of epsilon increase
        the quantile approximation, and hence result in more
        unequal buckets, but could improve performance
        and resource consumption.
    output_mode: Specification for the output of the layer.
        Values can be `"int"`, `"one_hot"`, `"multi_hot"`, or
        `"count"` configuring the layer as follows:
        - `"int"`: Return the discretized bin indices directly.
        - `"one_hot"`: Encodes each individual element in the
            input into an array the same size as `num_bins`,
            containing a 1 at the input's bin
            index. If the last dimension is size 1, will encode on that
            dimension.  If the last dimension is not size 1,
            will append a new dimension for the encoded output.
        - `"multi_hot"`: Encodes each sample in the input into a
            single array the same size as `num_bins`,
            containing a 1 for each bin index
            index present in the sample.
            Treats the last dimension as the sample
            dimension, if input shape is `(..., sample_length)`,
            output shape will be `(..., num_tokens)`.
        - `"count"`: As `"multi_hot"`, but the int array contains
            a count of the number of times the bin index appeared
            in the sample.
        Defaults to `"int"`.
    sparse: Boolean. Only applicable to `"one_hot"`, `"multi_hot"`,
        and `"count"` output modes. Only supported with TensorFlow
        backend. If `True`, returns a `SparseTensor` instead of
        a dense `Tensor`. Defaults to `False`.

Examples:

Discretize float values based on provided buckets.
>>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])
>>> layer = Discretization(bin_boundaries=[0., 1., 2.])
>>> layer(input)
array([[0, 2, 3, 1],
       [1, 3, 2, 1]])

Discretize float values based on a number of buckets to compute.
>>> input = np.array([[-1.5, 1.0, 3.4, .5], [0.0, 3.0, 1.3, 0.0]])
>>> layer = Discretization(num_bins=4, epsilon=0.01)
>>> layer.adapt(input)
>>> layer(input)
array([[0, 2, 3, 2],
       [1, 3, 3, 1]])
