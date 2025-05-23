% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/layers-preprocessing.R
\name{layer_random_grayscale}
\alias{layer_random_grayscale}
\title{Preprocessing layer for random conversion of RGB images to grayscale.}
\usage{
layer_random_grayscale(
  object,
  factor = 0.5,
  data_format = NULL,
  seed = NULL,
  ...
)
}
\arguments{
\item{object}{Object to compose the layer with. A tensor, array, or sequential model.}

\item{factor}{Float between 0 and 1, specifying the factor of
converting each image to grayscale. Defaults to 0.5. A value of
1.0 means all images will be converted, while 0.0 means no images
will be converted.}

\item{data_format}{String, one of \code{"channels_last"} (default) or
\code{"channels_first"}. The ordering of the dimensions in the inputs.
\code{"channels_last"} corresponds to inputs with shape
\verb{(batch, height, width, channels)} while \code{"channels_first"}
corresponds to inputs with shape
\verb{(batch, channels, height, width)}.}

\item{seed}{Initial seed for the random number generator}

\item{...}{For forward/backward compatability.}
}
\description{
This layer randomly converts input images to grayscale with a specified
factor. When applied, it maintains the original number of channels
but sets all channels to the same grayscale value. This can be useful
for data augmentation and training models to be robust to color
variations.

The conversion preserves the perceived luminance of the original color
image using standard RGB to grayscale conversion coefficients. Images
that are not selected for conversion remain unchanged.

\strong{Note:} This layer is safe to use inside a \code{tf.data} pipeline
(independently of which backend you're using).
}
\section{Input Shape}{
3D (unbatched) or 4D (batched) tensor with shape:
\verb{(..., height, width, channels)}, in \code{"channels_last"} format,
or \verb{(..., channels, height, width)}, in \code{"channels_first"} format.
}

\section{Output Shape}{
Same as input shape. The output maintains the same number of channels
as the input, even for grayscale-converted images where all channels
will have the same value.
}

\seealso{
Other image preprocessing layers: \cr
\code{\link{layer_aug_mix}()} \cr
\code{\link{layer_auto_contrast}()} \cr
\code{\link{layer_center_crop}()} \cr
\code{\link{layer_cut_mix}()} \cr
\code{\link{layer_equalization}()} \cr
\code{\link{layer_max_num_bounding_boxes}()} \cr
\code{\link{layer_mix_up}()} \cr
\code{\link{layer_rand_augment}()} \cr
\code{\link{layer_random_color_degeneration}()} \cr
\code{\link{layer_random_color_jitter}()} \cr
\code{\link{layer_random_erasing}()} \cr
\code{\link{layer_random_gaussian_blur}()} \cr
\code{\link{layer_random_hue}()} \cr
\code{\link{layer_random_invert}()} \cr
\code{\link{layer_random_perspective}()} \cr
\code{\link{layer_random_posterization}()} \cr
\code{\link{layer_random_saturation}()} \cr
\code{\link{layer_random_sharpness}()} \cr
\code{\link{layer_random_shear}()} \cr
\code{\link{layer_rescaling}()} \cr
\code{\link{layer_resizing}()} \cr
\code{\link{layer_solarization}()} \cr

Other preprocessing layers: \cr
\code{\link{layer_aug_mix}()} \cr
\code{\link{layer_auto_contrast}()} \cr
\code{\link{layer_category_encoding}()} \cr
\code{\link{layer_center_crop}()} \cr
\code{\link{layer_cut_mix}()} \cr
\code{\link{layer_discretization}()} \cr
\code{\link{layer_equalization}()} \cr
\code{\link{layer_feature_space}()} \cr
\code{\link{layer_hashed_crossing}()} \cr
\code{\link{layer_hashing}()} \cr
\code{\link{layer_integer_lookup}()} \cr
\code{\link{layer_max_num_bounding_boxes}()} \cr
\code{\link{layer_mel_spectrogram}()} \cr
\code{\link{layer_mix_up}()} \cr
\code{\link{layer_normalization}()} \cr
\code{\link{layer_rand_augment}()} \cr
\code{\link{layer_random_brightness}()} \cr
\code{\link{layer_random_color_degeneration}()} \cr
\code{\link{layer_random_color_jitter}()} \cr
\code{\link{layer_random_contrast}()} \cr
\code{\link{layer_random_crop}()} \cr
\code{\link{layer_random_erasing}()} \cr
\code{\link{layer_random_flip}()} \cr
\code{\link{layer_random_gaussian_blur}()} \cr
\code{\link{layer_random_hue}()} \cr
\code{\link{layer_random_invert}()} \cr
\code{\link{layer_random_perspective}()} \cr
\code{\link{layer_random_posterization}()} \cr
\code{\link{layer_random_rotation}()} \cr
\code{\link{layer_random_saturation}()} \cr
\code{\link{layer_random_sharpness}()} \cr
\code{\link{layer_random_shear}()} \cr
\code{\link{layer_random_translation}()} \cr
\code{\link{layer_random_zoom}()} \cr
\code{\link{layer_rescaling}()} \cr
\code{\link{layer_resizing}()} \cr
\code{\link{layer_solarization}()} \cr
\code{\link{layer_stft_spectrogram}()} \cr
\code{\link{layer_string_lookup}()} \cr
\code{\link{layer_text_vectorization}()} \cr

Other layers: \cr
\code{\link{Layer}()} \cr
\code{\link{layer_activation}()} \cr
\code{\link{layer_activation_elu}()} \cr
\code{\link{layer_activation_leaky_relu}()} \cr
\code{\link{layer_activation_parametric_relu}()} \cr
\code{\link{layer_activation_relu}()} \cr
\code{\link{layer_activation_softmax}()} \cr
\code{\link{layer_activity_regularization}()} \cr
\code{\link{layer_add}()} \cr
\code{\link{layer_additive_attention}()} \cr
\code{\link{layer_alpha_dropout}()} \cr
\code{\link{layer_attention}()} \cr
\code{\link{layer_aug_mix}()} \cr
\code{\link{layer_auto_contrast}()} \cr
\code{\link{layer_average}()} \cr
\code{\link{layer_average_pooling_1d}()} \cr
\code{\link{layer_average_pooling_2d}()} \cr
\code{\link{layer_average_pooling_3d}()} \cr
\code{\link{layer_batch_normalization}()} \cr
\code{\link{layer_bidirectional}()} \cr
\code{\link{layer_category_encoding}()} \cr
\code{\link{layer_center_crop}()} \cr
\code{\link{layer_concatenate}()} \cr
\code{\link{layer_conv_1d}()} \cr
\code{\link{layer_conv_1d_transpose}()} \cr
\code{\link{layer_conv_2d}()} \cr
\code{\link{layer_conv_2d_transpose}()} \cr
\code{\link{layer_conv_3d}()} \cr
\code{\link{layer_conv_3d_transpose}()} \cr
\code{\link{layer_conv_lstm_1d}()} \cr
\code{\link{layer_conv_lstm_2d}()} \cr
\code{\link{layer_conv_lstm_3d}()} \cr
\code{\link{layer_cropping_1d}()} \cr
\code{\link{layer_cropping_2d}()} \cr
\code{\link{layer_cropping_3d}()} \cr
\code{\link{layer_cut_mix}()} \cr
\code{\link{layer_dense}()} \cr
\code{\link{layer_depthwise_conv_1d}()} \cr
\code{\link{layer_depthwise_conv_2d}()} \cr
\code{\link{layer_discretization}()} \cr
\code{\link{layer_dot}()} \cr
\code{\link{layer_dropout}()} \cr
\code{\link{layer_einsum_dense}()} \cr
\code{\link{layer_embedding}()} \cr
\code{\link{layer_equalization}()} \cr
\code{\link{layer_feature_space}()} \cr
\code{\link{layer_flatten}()} \cr
\code{\link{layer_flax_module_wrapper}()} \cr
\code{\link{layer_gaussian_dropout}()} \cr
\code{\link{layer_gaussian_noise}()} \cr
\code{\link{layer_global_average_pooling_1d}()} \cr
\code{\link{layer_global_average_pooling_2d}()} \cr
\code{\link{layer_global_average_pooling_3d}()} \cr
\code{\link{layer_global_max_pooling_1d}()} \cr
\code{\link{layer_global_max_pooling_2d}()} \cr
\code{\link{layer_global_max_pooling_3d}()} \cr
\code{\link{layer_group_normalization}()} \cr
\code{\link{layer_group_query_attention}()} \cr
\code{\link{layer_gru}()} \cr
\code{\link{layer_hashed_crossing}()} \cr
\code{\link{layer_hashing}()} \cr
\code{\link{layer_identity}()} \cr
\code{\link{layer_integer_lookup}()} \cr
\code{\link{layer_jax_model_wrapper}()} \cr
\code{\link{layer_lambda}()} \cr
\code{\link{layer_layer_normalization}()} \cr
\code{\link{layer_lstm}()} \cr
\code{\link{layer_masking}()} \cr
\code{\link{layer_max_num_bounding_boxes}()} \cr
\code{\link{layer_max_pooling_1d}()} \cr
\code{\link{layer_max_pooling_2d}()} \cr
\code{\link{layer_max_pooling_3d}()} \cr
\code{\link{layer_maximum}()} \cr
\code{\link{layer_mel_spectrogram}()} \cr
\code{\link{layer_minimum}()} \cr
\code{\link{layer_mix_up}()} \cr
\code{\link{layer_multi_head_attention}()} \cr
\code{\link{layer_multiply}()} \cr
\code{\link{layer_normalization}()} \cr
\code{\link{layer_permute}()} \cr
\code{\link{layer_rand_augment}()} \cr
\code{\link{layer_random_brightness}()} \cr
\code{\link{layer_random_color_degeneration}()} \cr
\code{\link{layer_random_color_jitter}()} \cr
\code{\link{layer_random_contrast}()} \cr
\code{\link{layer_random_crop}()} \cr
\code{\link{layer_random_erasing}()} \cr
\code{\link{layer_random_flip}()} \cr
\code{\link{layer_random_gaussian_blur}()} \cr
\code{\link{layer_random_hue}()} \cr
\code{\link{layer_random_invert}()} \cr
\code{\link{layer_random_perspective}()} \cr
\code{\link{layer_random_posterization}()} \cr
\code{\link{layer_random_rotation}()} \cr
\code{\link{layer_random_saturation}()} \cr
\code{\link{layer_random_sharpness}()} \cr
\code{\link{layer_random_shear}()} \cr
\code{\link{layer_random_translation}()} \cr
\code{\link{layer_random_zoom}()} \cr
\code{\link{layer_repeat_vector}()} \cr
\code{\link{layer_rescaling}()} \cr
\code{\link{layer_reshape}()} \cr
\code{\link{layer_resizing}()} \cr
\code{\link{layer_rms_normalization}()} \cr
\code{\link{layer_rnn}()} \cr
\code{\link{layer_separable_conv_1d}()} \cr
\code{\link{layer_separable_conv_2d}()} \cr
\code{\link{layer_simple_rnn}()} \cr
\code{\link{layer_solarization}()} \cr
\code{\link{layer_spatial_dropout_1d}()} \cr
\code{\link{layer_spatial_dropout_2d}()} \cr
\code{\link{layer_spatial_dropout_3d}()} \cr
\code{\link{layer_spectral_normalization}()} \cr
\code{\link{layer_stft_spectrogram}()} \cr
\code{\link{layer_string_lookup}()} \cr
\code{\link{layer_subtract}()} \cr
\code{\link{layer_text_vectorization}()} \cr
\code{\link{layer_tfsm}()} \cr
\code{\link{layer_time_distributed}()} \cr
\code{\link{layer_torch_module_wrapper}()} \cr
\code{\link{layer_unit_normalization}()} \cr
\code{\link{layer_upsampling_1d}()} \cr
\code{\link{layer_upsampling_2d}()} \cr
\code{\link{layer_upsampling_3d}()} \cr
\code{\link{layer_zero_padding_1d}()} \cr
\code{\link{layer_zero_padding_2d}()} \cr
\code{\link{layer_zero_padding_3d}()} \cr
\code{\link{rnn_cell_gru}()} \cr
\code{\link{rnn_cell_lstm}()} \cr
\code{\link{rnn_cell_simple}()} \cr
\code{\link{rnn_cells_stack}()} \cr
}
\concept{image preprocessing layers}
\concept{layers}
\concept{preprocessing layers}
