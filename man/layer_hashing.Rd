% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/layers-preprocessing.R
\name{layer_hashing}
\alias{layer_hashing}
\title{A preprocessing layer which hashes and bins categorical features.}
\usage{
layer_hashing(
  object,
  num_bins,
  mask_value = NULL,
  salt = NULL,
  output_mode = "int",
  sparse = FALSE,
  ...
)
}
\arguments{
\item{object}{Object to compose the layer with. A tensor, array, or sequential model.}

\item{num_bins}{Number of hash bins. Note that this includes the \code{mask_value}
bin, so the effective number of bins is \code{(num_bins - 1)}
if \code{mask_value} is set.}

\item{mask_value}{A value that represents masked inputs, which are mapped to
index 0. \code{NULL} means no mask term will be added and the
hashing will start at index 0. Defaults to \code{NULL}.}

\item{salt}{A single unsigned integer or \code{NULL}.
If passed, the hash function used will be SipHash64,
with these values used as an additional input
(known as a "salt" in cryptography).
These should be non-zero. If \code{NULL}, uses the FarmHash64 hash
function. It also supports list of 2 unsigned
integer numbers, see reference paper for details.
Defaults to \code{NULL}.}

\item{output_mode}{Specification for the output of the layer. Values can be
\code{"int"}, \code{"one_hot"}, \code{"multi_hot"}, or
\code{"count"} configuring the layer as follows:
\itemize{
\item \code{"int"}: Return the integer bin indices directly.
\item \code{"one_hot"}: Encodes each individual element in the input into an
array the same size as \code{num_bins}, containing a 1
at the input's bin index. If the last dimension is size 1,
will encode on that dimension.
If the last dimension is not size 1, will append a new
dimension for the encoded output.
\item \code{"multi_hot"}: Encodes each sample in the input into a
single array the same size as \code{num_bins},
containing a 1 for each bin index
index present in the sample. Treats the last dimension
as the sample dimension, if input shape is
\verb{(..., sample_length)}, output shape will be
\verb{(..., num_tokens)}.
\item \code{"count"}: As \code{"multi_hot"}, but the int array contains a count of
the number of times the bin index appeared in the sample.
Defaults to \code{"int"}.
}}

\item{sparse}{Boolean. Only applicable to \code{"one_hot"}, \code{"multi_hot"},
and \code{"count"} output modes. Only supported with TensorFlow
backend. If \code{TRUE}, returns a \code{SparseTensor} instead of
a dense \code{Tensor}. Defaults to \code{FALSE}.}

\item{...}{Keyword arguments to construct a layer.}
}
\value{
The return value depends on the value provided for the first argument.
If  \code{object} is:
\itemize{
\item a \code{keras_model_sequential()}, then the layer is added to the sequential model
(which is modified in place). To enable piping, the sequential model is also
returned, invisibly.
\item a \code{keras_input()}, then the output tensor from calling \code{layer(input)} is returned.
\item \code{NULL} or missing, then a \code{Layer} instance is returned.
}
}
\description{
This layer transforms categorical inputs to hashed output. It element-wise
converts a ints or strings to ints in a fixed range. The stable hash
function uses \verb{tensorflow::ops::Fingerprint} to produce the same output
consistently across all platforms.

This layer uses \href{https://github.com/google/farmhash}{FarmHash64} by default,
which provides a consistent hashed output across different platforms and is
stable across invocations, regardless of device and context, by mixing the
input bits thoroughly.

If you want to obfuscate the hashed output, you can also pass a random
\code{salt} argument in the constructor. In that case, the layer will use the
\href{https://github.com/google/highwayhash}{SipHash64} hash function, with
the \code{salt} value serving as additional input to the hash function.

\strong{Note:} This layer internally uses TensorFlow. It cannot
be used as part of the compiled computation graph of a model with
any backend other than TensorFlow.
It can however be used with any backend when running eagerly.
It can also always be used as part of an input preprocessing pipeline
with any backend (outside the model itself), which is how we recommend
to use this layer.

\strong{Note:} This layer is safe to use inside a \code{tf.data} pipeline
(independently of which backend you're using).

\strong{Example (FarmHash64)}

\if{html}{\out{<div class="sourceCode r">}}\preformatted{layer <- layer_hashing(num_bins = 3)
inp <- c('A', 'B', 'C', 'D', 'E') |> array(dim = c(5, 1))
layer(inp)
}\if{html}{\out{</div>}}

\if{html}{\out{<div class="sourceCode">}}\preformatted{## tf.Tensor(
## [[1]
##  [0]
##  [1]
##  [1]
##  [2]], shape=(5, 1), dtype=int64)

}\if{html}{\out{</div>}}

\strong{Example (FarmHash64) with a mask value}

\if{html}{\out{<div class="sourceCode r">}}\preformatted{layer <- layer_hashing(num_bins=3, mask_value='')
inp <- c('A', 'B', '', 'C', 'D') |> array(dim = c(5, 1))
layer(inp)
}\if{html}{\out{</div>}}

\if{html}{\out{<div class="sourceCode">}}\preformatted{## tf.Tensor(
## [[1]
##  [1]
##  [0]
##  [2]
##  [2]], shape=(5, 1), dtype=int64)

}\if{html}{\out{</div>}}

\strong{Example (SipHash64)}

\if{html}{\out{<div class="sourceCode r">}}\preformatted{layer <- layer_hashing(num_bins=3, salt=c(133, 137))
inp <- c('A', 'B', 'C', 'D', 'E') |> array(dim = c(5, 1))
layer(inp)
}\if{html}{\out{</div>}}

\if{html}{\out{<div class="sourceCode">}}\preformatted{## tf.Tensor(
## [[1]
##  [2]
##  [1]
##  [0]
##  [2]], shape=(5, 1), dtype=int64)

}\if{html}{\out{</div>}}

\strong{Example (Siphash64 with a single integer, same as \verb{salt=[133, 133]})}

\if{html}{\out{<div class="sourceCode r">}}\preformatted{layer <- layer_hashing(num_bins=3, salt=133)
inp <- c('A', 'B', 'C', 'D', 'E') |> array(dim = c(5, 1))
layer(inp)
}\if{html}{\out{</div>}}

\if{html}{\out{<div class="sourceCode">}}\preformatted{## tf.Tensor(
## [[0]
##  [0]
##  [2]
##  [1]
##  [0]], shape=(5, 1), dtype=int64)

}\if{html}{\out{</div>}}
}
\section{Input Shape}{
A single string, a list of strings, or an \code{int32} or \code{int64} tensor
of shape \verb{(batch_size, ...,)}.
}

\section{Output Shape}{
An \code{int32} tensor of shape \verb{(batch_size, ...)}.
}

\section{Reference}{
\itemize{
\item \href{https://en.wikipedia.org/wiki/SipHash}{SipHash with salt}
}
}

\seealso{
\itemize{
\item \url{https://keras.io/api/layers/preprocessing_layers/categorical/hashing#hashing-class}
}

Other categorical features preprocessing layers: \cr
\code{\link{layer_category_encoding}()} \cr
\code{\link{layer_hashed_crossing}()} \cr
\code{\link{layer_integer_lookup}()} \cr
\code{\link{layer_string_lookup}()} \cr

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
\code{\link{layer_random_grayscale}()} \cr
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
\code{\link{layer_random_grayscale}()} \cr
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
\concept{categorical features preprocessing layers}
\concept{layers}
\concept{preprocessing layers}
