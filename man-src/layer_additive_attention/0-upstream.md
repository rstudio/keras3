keras.layers.AdditiveAttention
__signature__
(use_scale=True, dropout=0.0, **kwargs)
__doc__
Additive attention layer, a.k.a. Bahdanau-style attention.

Inputs are a list with 2 or 3 elements:
1. A `query` tensor of shape `(batch_size, Tq, dim)`.
2. A `value` tensor of shape `(batch_size, Tv, dim)`.
3. A optional `key` tensor of shape `(batch_size, Tv, dim)`. If none
    supplied, `value` will be used as `key`.

The calculation follows the steps:
1. Calculate attention scores using `query` and `key` with shape
    `(batch_size, Tq, Tv)` as a non-linear sum
    `scores = reduce_sum(tanh(query + key), axis=-1)`.
2. Use scores to calculate a softmax distribution with shape
    `(batch_size, Tq, Tv)`.
3. Use the softmax distribution to create a linear combination of `value`
    with shape `(batch_size, Tq, dim)`.

Args:
    use_scale: If `True`, will create a scalar variable to scale the
        attention scores.
    dropout: Float between 0 and 1. Fraction of the units to drop for the
        attention scores. Defaults to `0.0`.

Call Args:
    inputs: List of the following tensors:
        - `query`: Query tensor of shape `(batch_size, Tq, dim)`.
        - `value`: Value tensor of shape `(batch_size, Tv, dim)`.
        - `key`: Optional key tensor of shape `(batch_size, Tv, dim)`. If
            not given, will use `value` for both `key` and `value`, which is
            the most common case.
    mask: List of the following tensors:
        - `query_mask`: A boolean mask tensor of shape `(batch_size, Tq)`.
            If given, the output will be zero at the positions where
            `mask==False`.
        - `value_mask`: A boolean mask tensor of shape `(batch_size, Tv)`.
            If given, will apply the mask such that values at positions
             where `mask==False` do not contribute to the result.
    return_attention_scores: bool, it `True`, returns the attention scores
        (after masking and softmax) as an additional output argument.
    training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (no dropout).
    use_causal_mask: Boolean. Set to `True` for decoder self-attention. Adds
        a mask such that position `i` cannot attend to positions `j > i`.
        This prevents the flow of information from the future towards the
        past. Defaults to `False`.

Output:
    Attention outputs of shape `(batch_size, Tq, dim)`.
    (Optional) Attention scores after masking and softmax with shape
        `(batch_size, Tq, Tv)`.
