# Computes element-wise dot product of two tensors.

It takes a list of inputs of size 2, and the axes corresponding to each
input along with the dot product is to be performed.

Let's say `x` and `y` are the two input tensors with shapes `(2, 3, 5)`
and `(2, 10, 3)`. The batch dimension should be of same size for both
the inputs, and `axes` should correspond to the dimensions that have the
same size in the corresponding inputs. e.g. with `axes = c(1, 2)`, the
dot product of `x`, and `y` will result in a tensor with shape
`(2, 5, 10)`

## Usage

``` r
layer_dot(inputs, ..., axes, normalize = FALSE)
```

## Arguments

- inputs:

  layers to combine

- ...:

  Standard layer keyword arguments.

- axes:

  Integer or list of integers, axis or axes along which to take the dot
  product. If a list, should be two integers corresponding to the
  desired axis from the first input and the desired axis from the second
  input, respectively. Note that the size of the two selected axes must
  match.

- normalize:

  Whether to L2-normalize samples along the dot product axis before
  taking the dot product. If set to `TRUE`, then the output of the dot
  product is the cosine proximity between the two samples.

## Value

A tensor, the dot product of the samples from the inputs.

## Examples

    x <- op_reshape(0:9,   c(1, 5, 2))
    y <- op_reshape(10:19, c(1, 2, 5))
    layer_dot(x, y, axes=c(2, 3))

    ## tf.Tensor(
    ## [[[260 360]
    ##   [320 445]]], shape=(1, 2, 2), dtype=int32)

Usage in a Keras model:

    x1 <- op_reshape(0:9, c(5, 2)) |> layer_dense(8)
    x2 <- op_reshape(10:19, c(5, 2)) |> layer_dense(8)
    shape(x1)

    ## shape(5, 8)

    shape(x2)

    ## shape(5, 8)

    y <- layer_dot(x1, x2, axes=2)
    shape(y)

    ## shape(5, 1)

## See also

- <https://keras.io/api/layers/merging_layers/dot#dot-class>

Other merging layers:  
[`layer_add()`](https://keras3.posit.co/reference/layer_add.md)  
[`layer_average()`](https://keras3.posit.co/reference/layer_average.md)  
[`layer_concatenate()`](https://keras3.posit.co/reference/layer_concatenate.md)  
[`layer_maximum()`](https://keras3.posit.co/reference/layer_maximum.md)  
[`layer_minimum()`](https://keras3.posit.co/reference/layer_minimum.md)  
[`layer_multiply()`](https://keras3.posit.co/reference/layer_multiply.md)  
[`layer_subtract()`](https://keras3.posit.co/reference/layer_subtract.md)  

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
[`layer_group_query_attention()`](https://keras3.posit.co/reference/layer_group_query_attention.md)  
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
