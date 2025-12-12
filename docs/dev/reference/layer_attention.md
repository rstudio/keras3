# Dot-product attention layer, a.k.a. Luong-style attention.

Inputs are a list with 2 or 3 elements:

1.  A `query` tensor of shape `(batch_size, Tq, dim)`.

2.  A `value` tensor of shape `(batch_size, Tv, dim)`.

3.  A optional `key` tensor of shape `(batch_size, Tv, dim)`. If none
    supplied, `value` will be used as a `key`.

The calculation follows the steps:

1.  Calculate attention scores using `query` and `key` with shape
    `(batch_size, Tq, Tv)`.

2.  Use scores to calculate a softmax distribution with shape
    `(batch_size, Tq, Tv)`.

3.  Use the softmax distribution to create a linear combination of
    `value` with shape `(batch_size, Tq, dim)`.

## Usage

``` r
layer_attention(
  object,
  use_scale = FALSE,
  score_mode = "dot",
  dropout = 0,
  seed = NULL,
  ...
)
```

## Arguments

- object:

  Object to compose the layer with. A tensor, array, or sequential
  model.

- use_scale:

  If `TRUE`, will create a scalar variable to scale the attention
  scores.

- score_mode:

  Function to use to compute attention scores, one of
  `{"dot", "concat"}`. `"dot"` refers to the dot product between the
  query and key vectors. `"concat"` refers to the hyperbolic tangent of
  the concatenation of the `query` and `key` vectors.

- dropout:

  Float between 0 and 1. Fraction of the units to drop for the attention
  scores. Defaults to `0.0`.

- seed:

  An integer to use as random seed in case of `dropout`.

- ...:

  For forward/backward compatability.

## Value

The return value depends on the value provided for the first argument.
If `object` is:

- a
  [`keras_model_sequential()`](https://keras3.posit.co/dev/reference/keras_model_sequential.md),
  then the layer is added to the sequential model (which is modified in
  place). To enable piping, the sequential model is also returned,
  invisibly.

- a
  [`keras_input()`](https://keras3.posit.co/dev/reference/keras_input.md),
  then the output tensor from calling `layer(input)` is returned.

- `NULL` or missing, then a `Layer` instance is returned.

## Call Arguments

- `inputs`: List of the following tensors:

  - `query`: Query tensor of shape `(batch_size, Tq, dim)`.

  - `value`: Value tensor of shape `(batch_size, Tv, dim)`.

  - `key`: Optional key tensor of shape `(batch_size, Tv, dim)`. If not
    given, will use `value` for both `key` and `value`, which is the
    most common case.

- `mask`: List of the following tensors:

  - `query_mask`: A boolean mask tensor of shape `(batch_size, Tq)`. If
    given, the output will be zero at the positions where `mask==FALSE`.

  - `value_mask`: A boolean mask tensor of shape `(batch_size, Tv)`. If
    given, will apply the mask such that values at positions where
    `mask==FALSE` do not contribute to the result.

- `return_attention_scores`: bool, it `TRUE`, returns the attention
  scores (after masking and softmax) as an additional output argument.

- `training`: Python boolean indicating whether the layer should behave
  in training mode (adding dropout) or in inference mode (no dropout).

- `use_causal_mask`: Boolean. Set to `TRUE` for decoder self-attention.
  Adds a mask such that position `i` cannot attend to positions `j > i`.
  This prevents the flow of information from the future towards the
  past. Defaults to `FALSE`.

## Output

Attention outputs of shape `(batch_size, Tq, dim)`. (Optional) Attention
scores after masking and softmax with shape `(batch_size, Tq, Tv)`.

## See also

- <https://keras.io/api/layers/attention_layers/attention#attention-class>

Other attention layers:  
[`layer_additive_attention()`](https://keras3.posit.co/dev/reference/layer_additive_attention.md)  
[`layer_group_query_attention()`](https://keras3.posit.co/dev/reference/layer_group_query_attention.md)  
[`layer_multi_head_attention()`](https://keras3.posit.co/dev/reference/layer_multi_head_attention.md)  

Other layers:  
[`Layer()`](https://keras3.posit.co/dev/reference/Layer.md)  
[`layer_activation()`](https://keras3.posit.co/dev/reference/layer_activation.md)  
[`layer_activation_elu()`](https://keras3.posit.co/dev/reference/layer_activation_elu.md)  
[`layer_activation_leaky_relu()`](https://keras3.posit.co/dev/reference/layer_activation_leaky_relu.md)  
[`layer_activation_parametric_relu()`](https://keras3.posit.co/dev/reference/layer_activation_parametric_relu.md)  
[`layer_activation_relu()`](https://keras3.posit.co/dev/reference/layer_activation_relu.md)  
[`layer_activation_softmax()`](https://keras3.posit.co/dev/reference/layer_activation_softmax.md)  
[`layer_activity_regularization()`](https://keras3.posit.co/dev/reference/layer_activity_regularization.md)  
[`layer_add()`](https://keras3.posit.co/dev/reference/layer_add.md)  
[`layer_additive_attention()`](https://keras3.posit.co/dev/reference/layer_additive_attention.md)  
[`layer_alpha_dropout()`](https://keras3.posit.co/dev/reference/layer_alpha_dropout.md)  
[`layer_aug_mix()`](https://keras3.posit.co/dev/reference/layer_aug_mix.md)  
[`layer_auto_contrast()`](https://keras3.posit.co/dev/reference/layer_auto_contrast.md)  
[`layer_average()`](https://keras3.posit.co/dev/reference/layer_average.md)  
[`layer_average_pooling_1d()`](https://keras3.posit.co/dev/reference/layer_average_pooling_1d.md)  
[`layer_average_pooling_2d()`](https://keras3.posit.co/dev/reference/layer_average_pooling_2d.md)  
[`layer_average_pooling_3d()`](https://keras3.posit.co/dev/reference/layer_average_pooling_3d.md)  
[`layer_batch_normalization()`](https://keras3.posit.co/dev/reference/layer_batch_normalization.md)  
[`layer_bidirectional()`](https://keras3.posit.co/dev/reference/layer_bidirectional.md)  
[`layer_category_encoding()`](https://keras3.posit.co/dev/reference/layer_category_encoding.md)  
[`layer_center_crop()`](https://keras3.posit.co/dev/reference/layer_center_crop.md)  
[`layer_concatenate()`](https://keras3.posit.co/dev/reference/layer_concatenate.md)  
[`layer_conv_1d()`](https://keras3.posit.co/dev/reference/layer_conv_1d.md)  
[`layer_conv_1d_transpose()`](https://keras3.posit.co/dev/reference/layer_conv_1d_transpose.md)  
[`layer_conv_2d()`](https://keras3.posit.co/dev/reference/layer_conv_2d.md)  
[`layer_conv_2d_transpose()`](https://keras3.posit.co/dev/reference/layer_conv_2d_transpose.md)  
[`layer_conv_3d()`](https://keras3.posit.co/dev/reference/layer_conv_3d.md)  
[`layer_conv_3d_transpose()`](https://keras3.posit.co/dev/reference/layer_conv_3d_transpose.md)  
[`layer_conv_lstm_1d()`](https://keras3.posit.co/dev/reference/layer_conv_lstm_1d.md)  
[`layer_conv_lstm_2d()`](https://keras3.posit.co/dev/reference/layer_conv_lstm_2d.md)  
[`layer_conv_lstm_3d()`](https://keras3.posit.co/dev/reference/layer_conv_lstm_3d.md)  
[`layer_cropping_1d()`](https://keras3.posit.co/dev/reference/layer_cropping_1d.md)  
[`layer_cropping_2d()`](https://keras3.posit.co/dev/reference/layer_cropping_2d.md)  
[`layer_cropping_3d()`](https://keras3.posit.co/dev/reference/layer_cropping_3d.md)  
[`layer_cut_mix()`](https://keras3.posit.co/dev/reference/layer_cut_mix.md)  
[`layer_dense()`](https://keras3.posit.co/dev/reference/layer_dense.md)  
[`layer_depthwise_conv_1d()`](https://keras3.posit.co/dev/reference/layer_depthwise_conv_1d.md)  
[`layer_depthwise_conv_2d()`](https://keras3.posit.co/dev/reference/layer_depthwise_conv_2d.md)  
[`layer_discretization()`](https://keras3.posit.co/dev/reference/layer_discretization.md)  
[`layer_dot()`](https://keras3.posit.co/dev/reference/layer_dot.md)  
[`layer_dropout()`](https://keras3.posit.co/dev/reference/layer_dropout.md)  
[`layer_einsum_dense()`](https://keras3.posit.co/dev/reference/layer_einsum_dense.md)  
[`layer_embedding()`](https://keras3.posit.co/dev/reference/layer_embedding.md)  
[`layer_equalization()`](https://keras3.posit.co/dev/reference/layer_equalization.md)  
[`layer_feature_space()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)  
[`layer_flatten()`](https://keras3.posit.co/dev/reference/layer_flatten.md)  
[`layer_flax_module_wrapper()`](https://keras3.posit.co/dev/reference/layer_flax_module_wrapper.md)  
[`layer_gaussian_dropout()`](https://keras3.posit.co/dev/reference/layer_gaussian_dropout.md)  
[`layer_gaussian_noise()`](https://keras3.posit.co/dev/reference/layer_gaussian_noise.md)  
[`layer_global_average_pooling_1d()`](https://keras3.posit.co/dev/reference/layer_global_average_pooling_1d.md)  
[`layer_global_average_pooling_2d()`](https://keras3.posit.co/dev/reference/layer_global_average_pooling_2d.md)  
[`layer_global_average_pooling_3d()`](https://keras3.posit.co/dev/reference/layer_global_average_pooling_3d.md)  
[`layer_global_max_pooling_1d()`](https://keras3.posit.co/dev/reference/layer_global_max_pooling_1d.md)  
[`layer_global_max_pooling_2d()`](https://keras3.posit.co/dev/reference/layer_global_max_pooling_2d.md)  
[`layer_global_max_pooling_3d()`](https://keras3.posit.co/dev/reference/layer_global_max_pooling_3d.md)  
[`layer_group_normalization()`](https://keras3.posit.co/dev/reference/layer_group_normalization.md)  
[`layer_group_query_attention()`](https://keras3.posit.co/dev/reference/layer_group_query_attention.md)  
[`layer_gru()`](https://keras3.posit.co/dev/reference/layer_gru.md)  
[`layer_hashed_crossing()`](https://keras3.posit.co/dev/reference/layer_hashed_crossing.md)  
[`layer_hashing()`](https://keras3.posit.co/dev/reference/layer_hashing.md)  
[`layer_identity()`](https://keras3.posit.co/dev/reference/layer_identity.md)  
[`layer_integer_lookup()`](https://keras3.posit.co/dev/reference/layer_integer_lookup.md)  
[`layer_jax_model_wrapper()`](https://keras3.posit.co/dev/reference/layer_jax_model_wrapper.md)  
[`layer_lambda()`](https://keras3.posit.co/dev/reference/layer_lambda.md)  
[`layer_layer_normalization()`](https://keras3.posit.co/dev/reference/layer_layer_normalization.md)  
[`layer_lstm()`](https://keras3.posit.co/dev/reference/layer_lstm.md)  
[`layer_masking()`](https://keras3.posit.co/dev/reference/layer_masking.md)  
[`layer_max_num_bounding_boxes()`](https://keras3.posit.co/dev/reference/layer_max_num_bounding_boxes.md)  
[`layer_max_pooling_1d()`](https://keras3.posit.co/dev/reference/layer_max_pooling_1d.md)  
[`layer_max_pooling_2d()`](https://keras3.posit.co/dev/reference/layer_max_pooling_2d.md)  
[`layer_max_pooling_3d()`](https://keras3.posit.co/dev/reference/layer_max_pooling_3d.md)  
[`layer_maximum()`](https://keras3.posit.co/dev/reference/layer_maximum.md)  
[`layer_mel_spectrogram()`](https://keras3.posit.co/dev/reference/layer_mel_spectrogram.md)  
[`layer_minimum()`](https://keras3.posit.co/dev/reference/layer_minimum.md)  
[`layer_mix_up()`](https://keras3.posit.co/dev/reference/layer_mix_up.md)  
[`layer_multi_head_attention()`](https://keras3.posit.co/dev/reference/layer_multi_head_attention.md)  
[`layer_multiply()`](https://keras3.posit.co/dev/reference/layer_multiply.md)  
[`layer_normalization()`](https://keras3.posit.co/dev/reference/layer_normalization.md)  
[`layer_permute()`](https://keras3.posit.co/dev/reference/layer_permute.md)  
[`layer_rand_augment()`](https://keras3.posit.co/dev/reference/layer_rand_augment.md)  
[`layer_random_brightness()`](https://keras3.posit.co/dev/reference/layer_random_brightness.md)  
[`layer_random_color_degeneration()`](https://keras3.posit.co/dev/reference/layer_random_color_degeneration.md)  
[`layer_random_color_jitter()`](https://keras3.posit.co/dev/reference/layer_random_color_jitter.md)  
[`layer_random_contrast()`](https://keras3.posit.co/dev/reference/layer_random_contrast.md)  
[`layer_random_crop()`](https://keras3.posit.co/dev/reference/layer_random_crop.md)  
[`layer_random_elastic_transform()`](https://keras3.posit.co/dev/reference/layer_random_elastic_transform.md)  
[`layer_random_erasing()`](https://keras3.posit.co/dev/reference/layer_random_erasing.md)  
[`layer_random_flip()`](https://keras3.posit.co/dev/reference/layer_random_flip.md)  
[`layer_random_gaussian_blur()`](https://keras3.posit.co/dev/reference/layer_random_gaussian_blur.md)  
[`layer_random_grayscale()`](https://keras3.posit.co/dev/reference/layer_random_grayscale.md)  
[`layer_random_hue()`](https://keras3.posit.co/dev/reference/layer_random_hue.md)  
[`layer_random_invert()`](https://keras3.posit.co/dev/reference/layer_random_invert.md)  
[`layer_random_perspective()`](https://keras3.posit.co/dev/reference/layer_random_perspective.md)  
[`layer_random_posterization()`](https://keras3.posit.co/dev/reference/layer_random_posterization.md)  
[`layer_random_rotation()`](https://keras3.posit.co/dev/reference/layer_random_rotation.md)  
[`layer_random_saturation()`](https://keras3.posit.co/dev/reference/layer_random_saturation.md)  
[`layer_random_sharpness()`](https://keras3.posit.co/dev/reference/layer_random_sharpness.md)  
[`layer_random_shear()`](https://keras3.posit.co/dev/reference/layer_random_shear.md)  
[`layer_random_translation()`](https://keras3.posit.co/dev/reference/layer_random_translation.md)  
[`layer_random_zoom()`](https://keras3.posit.co/dev/reference/layer_random_zoom.md)  
[`layer_repeat_vector()`](https://keras3.posit.co/dev/reference/layer_repeat_vector.md)  
[`layer_rescaling()`](https://keras3.posit.co/dev/reference/layer_rescaling.md)  
[`layer_reshape()`](https://keras3.posit.co/dev/reference/layer_reshape.md)  
[`layer_resizing()`](https://keras3.posit.co/dev/reference/layer_resizing.md)  
[`layer_rms_normalization()`](https://keras3.posit.co/dev/reference/layer_rms_normalization.md)  
[`layer_rnn()`](https://keras3.posit.co/dev/reference/layer_rnn.md)  
[`layer_separable_conv_1d()`](https://keras3.posit.co/dev/reference/layer_separable_conv_1d.md)  
[`layer_separable_conv_2d()`](https://keras3.posit.co/dev/reference/layer_separable_conv_2d.md)  
[`layer_simple_rnn()`](https://keras3.posit.co/dev/reference/layer_simple_rnn.md)  
[`layer_solarization()`](https://keras3.posit.co/dev/reference/layer_solarization.md)  
[`layer_spatial_dropout_1d()`](https://keras3.posit.co/dev/reference/layer_spatial_dropout_1d.md)  
[`layer_spatial_dropout_2d()`](https://keras3.posit.co/dev/reference/layer_spatial_dropout_2d.md)  
[`layer_spatial_dropout_3d()`](https://keras3.posit.co/dev/reference/layer_spatial_dropout_3d.md)  
[`layer_spectral_normalization()`](https://keras3.posit.co/dev/reference/layer_spectral_normalization.md)  
[`layer_stft_spectrogram()`](https://keras3.posit.co/dev/reference/layer_stft_spectrogram.md)  
[`layer_string_lookup()`](https://keras3.posit.co/dev/reference/layer_string_lookup.md)  
[`layer_subtract()`](https://keras3.posit.co/dev/reference/layer_subtract.md)  
[`layer_text_vectorization()`](https://keras3.posit.co/dev/reference/layer_text_vectorization.md)  
[`layer_tfsm()`](https://keras3.posit.co/dev/reference/layer_tfsm.md)  
[`layer_time_distributed()`](https://keras3.posit.co/dev/reference/layer_time_distributed.md)  
[`layer_torch_module_wrapper()`](https://keras3.posit.co/dev/reference/layer_torch_module_wrapper.md)  
[`layer_unit_normalization()`](https://keras3.posit.co/dev/reference/layer_unit_normalization.md)  
[`layer_upsampling_1d()`](https://keras3.posit.co/dev/reference/layer_upsampling_1d.md)  
[`layer_upsampling_2d()`](https://keras3.posit.co/dev/reference/layer_upsampling_2d.md)  
[`layer_upsampling_3d()`](https://keras3.posit.co/dev/reference/layer_upsampling_3d.md)  
[`layer_zero_padding_1d()`](https://keras3.posit.co/dev/reference/layer_zero_padding_1d.md)  
[`layer_zero_padding_2d()`](https://keras3.posit.co/dev/reference/layer_zero_padding_2d.md)  
[`layer_zero_padding_3d()`](https://keras3.posit.co/dev/reference/layer_zero_padding_3d.md)  
[`rnn_cell_gru()`](https://keras3.posit.co/dev/reference/rnn_cell_gru.md)  
[`rnn_cell_lstm()`](https://keras3.posit.co/dev/reference/rnn_cell_lstm.md)  
[`rnn_cell_simple()`](https://keras3.posit.co/dev/reference/rnn_cell_simple.md)  
[`rnn_cells_stack()`](https://keras3.posit.co/dev/reference/rnn_cells_stack.md)  
