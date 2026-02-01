# Grouped Query Attention layer.

This is an implementation of grouped-query attention introduced by
[Ainslie et al., 2023](https://arxiv.org/abs/2305.13245). Here
`num_key_value_heads` denotes number of groups, setting
`num_key_value_heads` to 1 is equivalent to multi-query attention, and
when `num_key_value_heads` is equal to `num_query_heads` it is
equivalent to multi-head attention.

This layer first projects `query`, `key`, and `value` tensors. Then,
`key` and `value` are repeated to match the number of heads of `query`.

Then, the `query` is scaled and dot-producted with `key` tensors. These
are softmaxed to obtain attention probabilities. The value tensors are
then interpolated by these probabilities and concatenated back to a
single tensor.

## Usage

``` r
layer_group_query_attention(
  object,
  head_dim,
  num_query_heads,
  num_key_value_heads,
  dropout = 0,
  use_bias = TRUE,
  flash_attention = NULL,
  kernel_initializer = "glorot_uniform",
  bias_initializer = "zeros",
  kernel_regularizer = NULL,
  bias_regularizer = NULL,
  activity_regularizer = NULL,
  kernel_constraint = NULL,
  bias_constraint = NULL,
  seed = NULL,
  ...
)
```

## Arguments

- object:

  Object to compose the layer with. A tensor, array, or sequential
  model.

- head_dim:

  Size of each attention head.

- num_query_heads:

  Number of query attention heads.

- num_key_value_heads:

  Number of key and value attention heads.

- dropout:

  Dropout probability.

- use_bias:

  Boolean, whether the dense layers use bias vectors/matrices.

- flash_attention:

  If `NULL`, the layer attempts to use flash attention for faster and
  more memory-efficient attention computations when possible. This
  behavior can be configured using
  [`config_enable_flash_attention()`](https://keras3.posit.co/reference/config_enable_flash_attention.md)
  or
  [`config_disable_flash_attention()`](https://keras3.posit.co/reference/config_disable_flash_attention.md).

- kernel_initializer:

  Initializer for dense layer kernels.

- bias_initializer:

  Initializer for dense layer biases.

- kernel_regularizer:

  Regularizer for dense layer kernels.

- bias_regularizer:

  Regularizer for dense layer biases.

- activity_regularizer:

  Regularizer for dense layer activity.

- kernel_constraint:

  Constraint for dense layer kernels.

- bias_constraint:

  Constraint for dense layer kernels.

- seed:

  Optional integer to seed the dropout layer.

- ...:

  For forward/backward compatability.

## Value

attention_output: Result of the computation, of shape
`(batch_dim, target_seq_len, feature_dim)`, where `target_seq_len` is
for target sequence length and `feature_dim` is the query input last
dim. attention_scores: (Optional) attention coefficients of shape
`(batch_dim, num_query_heads, target_seq_len, source_seq_len)`.

## Call Arguments

- `query`: Query tensor of shape
  `(batch_dim, target_seq_len, feature_dim)`, where `batch_dim` is batch
  size, `target_seq_len` is the length of target sequence, and
  `feature_dim` is dimension of feature.

- `value`: Value tensor of shape
  `(batch_dim, source_seq_len, feature_dim)`, where `batch_dim` is batch
  size, `source_seq_len` is the length of source sequence, and
  `feature_dim` is dimension of feature.

- `key`: Optional key tensor of shape
  `(batch_dim, source_seq_len, feature_dim)`. If not given, will use
  `value` for both `key` and `value`, which is most common case.

- `attention_mask`: A boolean mask of shape
  `(batch_dim, target_seq_len, source_seq_len)`, that prevents attention
  to certain positions. The boolean mask specifies which query elements
  can attend to which key elements, where 1 indicates attention and 0
  indicates no attention. Broadcasting can happen for the missing batch
  dimensions and the head dimension.

- `return_attention_scores`: A boolean to indicate whether the output
  should be `(attention_output, attention_scores)` if `TRUE`, or
  `attention_output` if `FALSE`. Defaults to `FALSE`.

- `training`: Python boolean indicating whether the layer should behave
  in training mode (adding dropout) or in inference mode (no dropout).
  Will go with either using the training mode of the parent layer/model
  or `FALSE` (inference) if there is no parent layer.

- `use_causal_mask`: A boolean to indicate whether to apply a causal
  mask to prevent tokens from attending to future tokens (e.g., used in
  a decoder Transformer).

## See also

Other attention layers:  
[`layer_additive_attention()`](https://keras3.posit.co/reference/layer_additive_attention.md)  
[`layer_attention()`](https://keras3.posit.co/reference/layer_attention.md)  
[`layer_multi_head_attention()`](https://keras3.posit.co/reference/layer_multi_head_attention.md)  

Other layers:  
[`Layer()`](https://keras3.posit.co/reference/Layer.md)  
[`layer_activation()`](https://keras3.posit.co/reference/layer_activation.md)  
[`layer_activation_elu()`](https://keras3.posit.co/reference/layer_activation_elu.md)  
[`layer_activation_leaky_relu()`](https://keras3.posit.co/reference/layer_activation_leaky_relu.md)  
[`layer_activation_parametric_relu()`](https://keras3.posit.co/reference/layer_activation_parametric_relu.md)  
[`layer_activation_relu()`](https://keras3.posit.co/reference/layer_activation_relu.md)  
[`layer_activation_softmax()`](https://keras3.posit.co/reference/layer_activation_softmax.md)  
[`layer_activity_regularization()`](https://keras3.posit.co/reference/layer_activity_regularization.md)  
[`layer_add()`](https://keras3.posit.co/reference/layer_add.md)  
[`layer_additive_attention()`](https://keras3.posit.co/reference/layer_additive_attention.md)  
[`layer_alpha_dropout()`](https://keras3.posit.co/reference/layer_alpha_dropout.md)  
[`layer_attention()`](https://keras3.posit.co/reference/layer_attention.md)  
[`layer_aug_mix()`](https://keras3.posit.co/reference/layer_aug_mix.md)  
[`layer_auto_contrast()`](https://keras3.posit.co/reference/layer_auto_contrast.md)  
[`layer_average()`](https://keras3.posit.co/reference/layer_average.md)  
[`layer_average_pooling_1d()`](https://keras3.posit.co/reference/layer_average_pooling_1d.md)  
[`layer_average_pooling_2d()`](https://keras3.posit.co/reference/layer_average_pooling_2d.md)  
[`layer_average_pooling_3d()`](https://keras3.posit.co/reference/layer_average_pooling_3d.md)  
[`layer_batch_normalization()`](https://keras3.posit.co/reference/layer_batch_normalization.md)  
[`layer_bidirectional()`](https://keras3.posit.co/reference/layer_bidirectional.md)  
[`layer_category_encoding()`](https://keras3.posit.co/reference/layer_category_encoding.md)  
[`layer_center_crop()`](https://keras3.posit.co/reference/layer_center_crop.md)  
[`layer_concatenate()`](https://keras3.posit.co/reference/layer_concatenate.md)  
[`layer_conv_1d()`](https://keras3.posit.co/reference/layer_conv_1d.md)  
[`layer_conv_1d_transpose()`](https://keras3.posit.co/reference/layer_conv_1d_transpose.md)  
[`layer_conv_2d()`](https://keras3.posit.co/reference/layer_conv_2d.md)  
[`layer_conv_2d_transpose()`](https://keras3.posit.co/reference/layer_conv_2d_transpose.md)  
[`layer_conv_3d()`](https://keras3.posit.co/reference/layer_conv_3d.md)  
[`layer_conv_3d_transpose()`](https://keras3.posit.co/reference/layer_conv_3d_transpose.md)  
[`layer_conv_lstm_1d()`](https://keras3.posit.co/reference/layer_conv_lstm_1d.md)  
[`layer_conv_lstm_2d()`](https://keras3.posit.co/reference/layer_conv_lstm_2d.md)  
[`layer_conv_lstm_3d()`](https://keras3.posit.co/reference/layer_conv_lstm_3d.md)  
[`layer_cropping_1d()`](https://keras3.posit.co/reference/layer_cropping_1d.md)  
[`layer_cropping_2d()`](https://keras3.posit.co/reference/layer_cropping_2d.md)  
[`layer_cropping_3d()`](https://keras3.posit.co/reference/layer_cropping_3d.md)  
[`layer_cut_mix()`](https://keras3.posit.co/reference/layer_cut_mix.md)  
[`layer_dense()`](https://keras3.posit.co/reference/layer_dense.md)  
[`layer_depthwise_conv_1d()`](https://keras3.posit.co/reference/layer_depthwise_conv_1d.md)  
[`layer_depthwise_conv_2d()`](https://keras3.posit.co/reference/layer_depthwise_conv_2d.md)  
[`layer_discretization()`](https://keras3.posit.co/reference/layer_discretization.md)  
[`layer_dot()`](https://keras3.posit.co/reference/layer_dot.md)  
[`layer_dropout()`](https://keras3.posit.co/reference/layer_dropout.md)  
[`layer_einsum_dense()`](https://keras3.posit.co/reference/layer_einsum_dense.md)  
[`layer_embedding()`](https://keras3.posit.co/reference/layer_embedding.md)  
[`layer_equalization()`](https://keras3.posit.co/reference/layer_equalization.md)  
[`layer_feature_space()`](https://keras3.posit.co/reference/layer_feature_space.md)  
[`layer_flatten()`](https://keras3.posit.co/reference/layer_flatten.md)  
[`layer_flax_module_wrapper()`](https://keras3.posit.co/reference/layer_flax_module_wrapper.md)  
[`layer_gaussian_dropout()`](https://keras3.posit.co/reference/layer_gaussian_dropout.md)  
[`layer_gaussian_noise()`](https://keras3.posit.co/reference/layer_gaussian_noise.md)  
[`layer_global_average_pooling_1d()`](https://keras3.posit.co/reference/layer_global_average_pooling_1d.md)  
[`layer_global_average_pooling_2d()`](https://keras3.posit.co/reference/layer_global_average_pooling_2d.md)  
[`layer_global_average_pooling_3d()`](https://keras3.posit.co/reference/layer_global_average_pooling_3d.md)  
[`layer_global_max_pooling_1d()`](https://keras3.posit.co/reference/layer_global_max_pooling_1d.md)  
[`layer_global_max_pooling_2d()`](https://keras3.posit.co/reference/layer_global_max_pooling_2d.md)  
[`layer_global_max_pooling_3d()`](https://keras3.posit.co/reference/layer_global_max_pooling_3d.md)  
[`layer_group_normalization()`](https://keras3.posit.co/reference/layer_group_normalization.md)  
[`layer_gru()`](https://keras3.posit.co/reference/layer_gru.md)  
[`layer_hashed_crossing()`](https://keras3.posit.co/reference/layer_hashed_crossing.md)  
[`layer_hashing()`](https://keras3.posit.co/reference/layer_hashing.md)  
[`layer_identity()`](https://keras3.posit.co/reference/layer_identity.md)  
[`layer_integer_lookup()`](https://keras3.posit.co/reference/layer_integer_lookup.md)  
[`layer_jax_model_wrapper()`](https://keras3.posit.co/reference/layer_jax_model_wrapper.md)  
[`layer_lambda()`](https://keras3.posit.co/reference/layer_lambda.md)  
[`layer_layer_normalization()`](https://keras3.posit.co/reference/layer_layer_normalization.md)  
[`layer_lstm()`](https://keras3.posit.co/reference/layer_lstm.md)  
[`layer_masking()`](https://keras3.posit.co/reference/layer_masking.md)  
[`layer_max_num_bounding_boxes()`](https://keras3.posit.co/reference/layer_max_num_bounding_boxes.md)  
[`layer_max_pooling_1d()`](https://keras3.posit.co/reference/layer_max_pooling_1d.md)  
[`layer_max_pooling_2d()`](https://keras3.posit.co/reference/layer_max_pooling_2d.md)  
[`layer_max_pooling_3d()`](https://keras3.posit.co/reference/layer_max_pooling_3d.md)  
[`layer_maximum()`](https://keras3.posit.co/reference/layer_maximum.md)  
[`layer_mel_spectrogram()`](https://keras3.posit.co/reference/layer_mel_spectrogram.md)  
[`layer_minimum()`](https://keras3.posit.co/reference/layer_minimum.md)  
[`layer_mix_up()`](https://keras3.posit.co/reference/layer_mix_up.md)  
[`layer_multi_head_attention()`](https://keras3.posit.co/reference/layer_multi_head_attention.md)  
[`layer_multiply()`](https://keras3.posit.co/reference/layer_multiply.md)  
[`layer_normalization()`](https://keras3.posit.co/reference/layer_normalization.md)  
[`layer_permute()`](https://keras3.posit.co/reference/layer_permute.md)  
[`layer_rand_augment()`](https://keras3.posit.co/reference/layer_rand_augment.md)  
[`layer_random_brightness()`](https://keras3.posit.co/reference/layer_random_brightness.md)  
[`layer_random_color_degeneration()`](https://keras3.posit.co/reference/layer_random_color_degeneration.md)  
[`layer_random_color_jitter()`](https://keras3.posit.co/reference/layer_random_color_jitter.md)  
[`layer_random_contrast()`](https://keras3.posit.co/reference/layer_random_contrast.md)  
[`layer_random_crop()`](https://keras3.posit.co/reference/layer_random_crop.md)  
[`layer_random_elastic_transform()`](https://keras3.posit.co/reference/layer_random_elastic_transform.md)  
[`layer_random_erasing()`](https://keras3.posit.co/reference/layer_random_erasing.md)  
[`layer_random_flip()`](https://keras3.posit.co/reference/layer_random_flip.md)  
[`layer_random_gaussian_blur()`](https://keras3.posit.co/reference/layer_random_gaussian_blur.md)  
[`layer_random_grayscale()`](https://keras3.posit.co/reference/layer_random_grayscale.md)  
[`layer_random_hue()`](https://keras3.posit.co/reference/layer_random_hue.md)  
[`layer_random_invert()`](https://keras3.posit.co/reference/layer_random_invert.md)  
[`layer_random_perspective()`](https://keras3.posit.co/reference/layer_random_perspective.md)  
[`layer_random_posterization()`](https://keras3.posit.co/reference/layer_random_posterization.md)  
[`layer_random_rotation()`](https://keras3.posit.co/reference/layer_random_rotation.md)  
[`layer_random_saturation()`](https://keras3.posit.co/reference/layer_random_saturation.md)  
[`layer_random_sharpness()`](https://keras3.posit.co/reference/layer_random_sharpness.md)  
[`layer_random_shear()`](https://keras3.posit.co/reference/layer_random_shear.md)  
[`layer_random_translation()`](https://keras3.posit.co/reference/layer_random_translation.md)  
[`layer_random_zoom()`](https://keras3.posit.co/reference/layer_random_zoom.md)  
[`layer_repeat_vector()`](https://keras3.posit.co/reference/layer_repeat_vector.md)  
[`layer_rescaling()`](https://keras3.posit.co/reference/layer_rescaling.md)  
[`layer_reshape()`](https://keras3.posit.co/reference/layer_reshape.md)  
[`layer_resizing()`](https://keras3.posit.co/reference/layer_resizing.md)  
[`layer_rms_normalization()`](https://keras3.posit.co/reference/layer_rms_normalization.md)  
[`layer_rnn()`](https://keras3.posit.co/reference/layer_rnn.md)  
[`layer_separable_conv_1d()`](https://keras3.posit.co/reference/layer_separable_conv_1d.md)  
[`layer_separable_conv_2d()`](https://keras3.posit.co/reference/layer_separable_conv_2d.md)  
[`layer_simple_rnn()`](https://keras3.posit.co/reference/layer_simple_rnn.md)  
[`layer_solarization()`](https://keras3.posit.co/reference/layer_solarization.md)  
[`layer_spatial_dropout_1d()`](https://keras3.posit.co/reference/layer_spatial_dropout_1d.md)  
[`layer_spatial_dropout_2d()`](https://keras3.posit.co/reference/layer_spatial_dropout_2d.md)  
[`layer_spatial_dropout_3d()`](https://keras3.posit.co/reference/layer_spatial_dropout_3d.md)  
[`layer_spectral_normalization()`](https://keras3.posit.co/reference/layer_spectral_normalization.md)  
[`layer_stft_spectrogram()`](https://keras3.posit.co/reference/layer_stft_spectrogram.md)  
[`layer_string_lookup()`](https://keras3.posit.co/reference/layer_string_lookup.md)  
[`layer_subtract()`](https://keras3.posit.co/reference/layer_subtract.md)  
[`layer_text_vectorization()`](https://keras3.posit.co/reference/layer_text_vectorization.md)  
[`layer_tfsm()`](https://keras3.posit.co/reference/layer_tfsm.md)  
[`layer_time_distributed()`](https://keras3.posit.co/reference/layer_time_distributed.md)  
[`layer_torch_module_wrapper()`](https://keras3.posit.co/reference/layer_torch_module_wrapper.md)  
[`layer_unit_normalization()`](https://keras3.posit.co/reference/layer_unit_normalization.md)  
[`layer_upsampling_1d()`](https://keras3.posit.co/reference/layer_upsampling_1d.md)  
[`layer_upsampling_2d()`](https://keras3.posit.co/reference/layer_upsampling_2d.md)  
[`layer_upsampling_3d()`](https://keras3.posit.co/reference/layer_upsampling_3d.md)  
[`layer_zero_padding_1d()`](https://keras3.posit.co/reference/layer_zero_padding_1d.md)  
[`layer_zero_padding_2d()`](https://keras3.posit.co/reference/layer_zero_padding_2d.md)  
[`layer_zero_padding_3d()`](https://keras3.posit.co/reference/layer_zero_padding_3d.md)  
[`rnn_cell_gru()`](https://keras3.posit.co/reference/rnn_cell_gru.md)  
[`rnn_cell_lstm()`](https://keras3.posit.co/reference/rnn_cell_lstm.md)  
[`rnn_cell_simple()`](https://keras3.posit.co/reference/rnn_cell_simple.md)  
[`rnn_cells_stack()`](https://keras3.posit.co/reference/rnn_cells_stack.md)  
