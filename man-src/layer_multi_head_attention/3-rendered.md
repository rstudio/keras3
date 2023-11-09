MultiHeadAttention layer.

@description
This is an implementation of multi-headed attention as described in the
paper "Attention is all you Need"
[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).
If `query`, `key,` `value` are the same, then
this is self-attention. Each timestep in `query` attends to the
corresponding sequence in `key`, and returns a fixed-width vector.

This layer first projects `query`, `key` and `value`. These are
(effectively) a list of tensors of length `num_attention_heads`, where the
corresponding shapes are `(batch_size, <query dimensions>, key_dim)`,
`(batch_size, <key/value dimensions>, key_dim)`,
`(batch_size, <key/value dimensions>, value_dim)`.

Then, the query and key tensors are dot-producted and scaled. These are
softmaxed to obtain attention probabilities. The value tensors are then
interpolated by these probabilities, then concatenated back to a single
tensor.

Finally, the result tensor with the last dimension as `value_dim` can take
a linear projection and return.

# Call Arguments
- `query`: Query tensor of shape `(B, T, dim)`, where `B` is the batch size,
    `T` is the target sequence length, and dim is the feature dimension.
- `value`: Value tensor of shape `(B, S, dim)`, where `B` is the batch size,
    `S` is the source sequence length, and dim is the feature dimension.
- `key`: Optional key tensor of shape `(B, S, dim)`. If not given, will
    use `value` for both `key` and `value`, which is the most common
    case.
- `attention_mask`: a boolean mask of shape `(B, T, S)`, that prevents
    attention to certain positions. The boolean mask specifies which
    query elements can attend to which key elements, 1 indicates
    attention and 0 indicates no attention. Broadcasting can happen for
    the missing batch dimensions and the head dimension.
- `return_attention_scores`: A boolean to indicate whether the output should
    be `(attention_output, attention_scores)` if `True`, or
    `attention_output` if `False`. Defaults to `False`.
- `training`: Python boolean indicating whether the layer should behave in
    training mode (adding dropout) or in inference mode (no dropout).
    Will go with either using the training mode of the parent
    layer/model, or `False` (inference) if there is no parent layer.
- `use_causal_mask`: A boolean to indicate whether to apply a causal mask to
    prevent tokens from attending to future tokens (e.g., used in a
    decoder Transformer).

@returns
attention_output: The result of the computation, of shape `(B, T, E)`,
    where `T` is for target sequence shapes and `E` is the query input
    last dimension if `output_shape` is `None`. Otherwise, the
    multi-head outputs are projected to the shape specified by
    `output_shape`.
attention_scores: (Optional) multi-head attention coefficients over
    attention axes.

@param num_heads Number of attention heads.
@param key_dim Size of each attention head for query and key.
@param value_dim Size of each attention head for value.
@param dropout Dropout probability.
@param use_bias Boolean, whether the dense layers use bias vectors/matrices.
@param output_shape The expected shape of an output tensor, besides the batch
    and sequence dims. If not specified, projects back to the query
    feature dim (the query input's last dimension).
@param attention_axes axes over which the attention is applied. `None` means
    attention over all axes, but batch, heads, and features.
@param kernel_initializer Initializer for dense layer kernels.
@param bias_initializer Initializer for dense layer biases.
@param kernel_regularizer Regularizer for dense layer kernels.
@param bias_regularizer Regularizer for dense layer biases.
@param activity_regularizer Regularizer for dense layer activity.
@param kernel_constraint Constraint for dense layer kernels.
@param bias_constraint Constraint for dense layer kernels.
@param ... Passed on to the Python callable
@param inputs see description

@export
@family attention layers
@seealso
+ <https:/keras.io/api/layers/attention_layers/multi_head_attention#multiheadattention-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention>
