# Group normalization layer.

Group Normalization divides the channels into groups and computes within
each group the mean and variance for normalization. Empirically, its
accuracy is more stable than batch norm in a wide range of small batch
sizes, if learning rate is adjusted linearly with batch sizes.

Relation to Layer Normalization: If the number of groups is set to 1,
then this operation becomes nearly identical to Layer Normalization (see
Layer Normalization docs for details).

Relation to Instance Normalization: If the number of groups is set to
the input dimension (number of groups is equal to number of channels),
then this operation becomes identical to Instance Normalization. You can
achieve this via `groups=-1`.

## Usage

``` r
layer_group_normalization(
  object,
  groups = 32L,
  axis = -1L,
  epsilon = 0.001,
  center = TRUE,
  scale = TRUE,
  beta_initializer = "zeros",
  gamma_initializer = "ones",
  beta_regularizer = NULL,
  gamma_regularizer = NULL,
  beta_constraint = NULL,
  gamma_constraint = NULL,
  ...
)
```

## Arguments

- object:

  Object to compose the layer with. A tensor, array, or sequential
  model.

- groups:

  Integer, the number of groups for Group Normalization. Can be in the
  range `[1, N]` where N is the input dimension. The input dimension
  must be divisible by the number of groups. Defaults to 32.

- axis:

  Integer or List/Tuple. The axis or axes to normalize across.
  Typically, this is the features axis/axes. The left-out axes are
  typically the batch axis/axes. -1 is the last dimension in the input.
  Defaults to `-1`.

- epsilon:

  Small float added to variance to avoid dividing by zero. Defaults to
  1e-3.

- center:

  If `TRUE`, add offset of `beta` to normalized tensor. If `FALSE`,
  `beta` is ignored. Defaults to `TRUE`.

- scale:

  If `TRUE`, multiply by `gamma`. If `FALSE`, `gamma` is not used. When
  the next layer is linear (also e.g. `relu`), this can be disabled
  since the scaling will be done by the next layer. Defaults to `TRUE`.

- beta_initializer:

  Initializer for the beta weight. Defaults to zeros.

- gamma_initializer:

  Initializer for the gamma weight. Defaults to ones.

- beta_regularizer:

  Optional regularizer for the beta weight. `NULL` by default.

- gamma_regularizer:

  Optional regularizer for the gamma weight. `NULL` by default.

- beta_constraint:

  Optional constraint for the beta weight. `NULL` by default.

- gamma_constraint:

  Optional constraint for the gamma weight. `NULL` by default.

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

## Input Shape

Arbitrary. Use the keyword argument `input_shape` (tuple of integers,
does not include the samples axis) when using this layer as the first
layer in a model.

## Output Shape

Same shape as input. \*\*kwargs: Base layer keyword arguments (e.g.
`name` and `dtype`).

## Reference

- [Yuxin Wu & Kaiming He, 2018](https://arxiv.org/abs/1803.08494)

## See also

- <https://keras.io/api/layers/normalization_layers/group_normalization#groupnormalization-class>

Other normalization layers:  
[`layer_batch_normalization()`](https://keras3.posit.co/dev/reference/layer_batch_normalization.md)  
[`layer_layer_normalization()`](https://keras3.posit.co/dev/reference/layer_layer_normalization.md)  
[`layer_rms_normalization()`](https://keras3.posit.co/dev/reference/layer_rms_normalization.md)  
[`layer_spectral_normalization()`](https://keras3.posit.co/dev/reference/layer_spectral_normalization.md)  
[`layer_unit_normalization()`](https://keras3.posit.co/dev/reference/layer_unit_normalization.md)  

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
[`layer_attention()`](https://keras3.posit.co/dev/reference/layer_attention.md)  
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
