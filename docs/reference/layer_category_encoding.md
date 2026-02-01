# A preprocessing layer which encodes integer features.

This layer provides options for condensing data into a categorical
encoding when the total number of tokens are known in advance. It
accepts integer values as inputs, and it outputs a dense or sparse
representation of those inputs. For integer inputs where the total
number of tokens is not known, use
[`layer_integer_lookup()`](https://keras3.posit.co/reference/layer_integer_lookup.md)
instead.

**Note:** This layer is safe to use inside a `tf.data` pipeline
(independently of which backend you're using).

## Usage

``` r
layer_category_encoding(
  object,
  num_tokens = NULL,
  output_mode = "multi_hot",
  sparse = FALSE,
  ...
)
```

## Arguments

- object:

  Object to compose the layer with. A tensor, array, or sequential
  model.

- num_tokens:

  The total number of tokens the layer should support. All inputs to the
  layer must integers in the range `0 <= value < num_tokens`, or an
  error will be thrown.

- output_mode:

  Specification for the output of the layer. Values can be `"one_hot"`,
  `"multi_hot"` or `"count"`, configuring the layer as follows: -
  `"one_hot"`: Encodes each individual element in the input into an
  array of `num_tokens` size, containing a 1 at the element index. If
  the last dimension is size 1, will encode on that dimension. If the
  last dimension is not size 1, will append a new dimension for the
  encoded output. - `"multi_hot"`: Encodes each sample in the input into
  a single array of `num_tokens` size, containing a 1 for each
  vocabulary term present in the sample. Treats the last dimension as
  the sample dimension, if input shape is `(..., sample_length)`, output
  shape will be `(..., num_tokens)`. - `"count"`: Like `"multi_hot"`,
  but the int array contains a count of the number of times the token at
  that index appeared in the sample. For all output modes, currently
  only output up to rank 2 is supported. Defaults to `"multi_hot"`.

- sparse:

  Whether to return a sparse tensor; for backends that support sparse
  tensors.

- ...:

  For forward/backward compatability.

## Value

The return value depends on the value provided for the first argument.
If `object` is:

- a
  [`keras_model_sequential()`](https://keras3.posit.co/reference/keras_model_sequential.md),
  then the layer is added to the sequential model (which is modified in
  place). To enable piping, the sequential model is also returned,
  invisibly.

- a [`keras_input()`](https://keras3.posit.co/reference/keras_input.md),
  then the output tensor from calling `layer(input)` is returned.

- `NULL` or missing, then a `Layer` instance is returned.

## Examples

**One-hot encoding data**

    layer <- layer_category_encoding(num_tokens = 4, output_mode = "one_hot")
    x <- op_array(c(3, 2, 0, 1), "int32")
    layer(x)

    ## tf.Tensor(
    ## [[0. 0. 0. 1.]
    ##  [0. 0. 1. 0.]
    ##  [1. 0. 0. 0.]
    ##  [0. 1. 0. 0.]], shape=(4, 4), dtype=float32)

**Multi-hot encoding data**

    layer <- layer_category_encoding(num_tokens = 4, output_mode = "multi_hot")
    x <- op_array(rbind(c(0, 1),
                       c(0, 0),
                       c(1, 2),
                       c(3, 1)), "int32")
    layer(x)

    ## tf.Tensor(
    ## [[1. 1. 0. 0.]
    ##  [1. 0. 0. 0.]
    ##  [0. 1. 1. 0.]
    ##  [0. 1. 0. 1.]], shape=(4, 4), dtype=float32)

**Using weighted inputs in `"count"` mode**

    layer <- layer_category_encoding(num_tokens = 4, output_mode = "count")
    count_weights <- op_array(rbind(c(.1, .2),
                                   c(.1, .1),
                                   c(.2, .3),
                                   c(.4, .2)))
    x <- op_array(rbind(c(0, 1),
                       c(0, 0),
                       c(1, 2),
                       c(3, 1)), "int32")
    layer(x, count_weights = count_weights)
    #   array([[01, 02, 0. , 0. ],
    #          [02, 0. , 0. , 0. ],
    #          [0. , 02, 03, 0. ],
    #          [0. , 02, 0. , 04]]>

## Call Arguments

- `inputs`: A 1D or 2D tensor of integer inputs.

- `count_weights`: A tensor in the same shape as `inputs` indicating the
  weight for each sample value when summing up in `count` mode. Not used
  in `"multi_hot"` or `"one_hot"` modes.

## See also

- <https://keras.io/api/layers/preprocessing_layers/categorical/category_encoding#categoryencoding-class>

Other categorical features preprocessing layers:  
[`layer_hashed_crossing()`](https://keras3.posit.co/reference/layer_hashed_crossing.md)  
[`layer_hashing()`](https://keras3.posit.co/reference/layer_hashing.md)  
[`layer_integer_lookup()`](https://keras3.posit.co/reference/layer_integer_lookup.md)  
[`layer_string_lookup()`](https://keras3.posit.co/reference/layer_string_lookup.md)  

Other preprocessing layers:  
[`layer_aug_mix()`](https://keras3.posit.co/reference/layer_aug_mix.md)  
[`layer_auto_contrast()`](https://keras3.posit.co/reference/layer_auto_contrast.md)  
[`layer_center_crop()`](https://keras3.posit.co/reference/layer_center_crop.md)  
[`layer_cut_mix()`](https://keras3.posit.co/reference/layer_cut_mix.md)  
[`layer_discretization()`](https://keras3.posit.co/reference/layer_discretization.md)  
[`layer_equalization()`](https://keras3.posit.co/reference/layer_equalization.md)  
[`layer_feature_space()`](https://keras3.posit.co/reference/layer_feature_space.md)  
[`layer_hashed_crossing()`](https://keras3.posit.co/reference/layer_hashed_crossing.md)  
[`layer_hashing()`](https://keras3.posit.co/reference/layer_hashing.md)  
[`layer_integer_lookup()`](https://keras3.posit.co/reference/layer_integer_lookup.md)  
[`layer_max_num_bounding_boxes()`](https://keras3.posit.co/reference/layer_max_num_bounding_boxes.md)  
[`layer_mel_spectrogram()`](https://keras3.posit.co/reference/layer_mel_spectrogram.md)  
[`layer_mix_up()`](https://keras3.posit.co/reference/layer_mix_up.md)  
[`layer_normalization()`](https://keras3.posit.co/reference/layer_normalization.md)  
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
[`layer_rescaling()`](https://keras3.posit.co/reference/layer_rescaling.md)  
[`layer_resizing()`](https://keras3.posit.co/reference/layer_resizing.md)  
[`layer_solarization()`](https://keras3.posit.co/reference/layer_solarization.md)  
[`layer_stft_spectrogram()`](https://keras3.posit.co/reference/layer_stft_spectrogram.md)  
[`layer_string_lookup()`](https://keras3.posit.co/reference/layer_string_lookup.md)  
[`layer_text_vectorization()`](https://keras3.posit.co/reference/layer_text_vectorization.md)  

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
