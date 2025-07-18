% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/model-persistence.R
\name{layer_tfsm}
\alias{layer_tfsm}
\title{Reload a Keras model/layer that was saved via \code{export_savedmodel()}.}
\usage{
layer_tfsm(
  object,
  filepath,
  call_endpoint = "serve",
  call_training_endpoint = NULL,
  trainable = TRUE,
  name = NULL,
  dtype = NULL
)
}
\arguments{
\item{object}{Object to compose the layer with. A tensor, array, or sequential model.}

\item{filepath}{string, the path to the SavedModel.}

\item{call_endpoint}{Name of the endpoint to use as the \code{call()} method
of the reloaded layer. If the SavedModel was created
via \code{export_savedmodel()},
then the default endpoint name is \code{'serve'}. In other cases
it may be named \code{'serving_default'}.}

\item{call_training_endpoint}{see description}

\item{trainable}{see description}

\item{name}{String, name for the object}

\item{dtype}{datatype (e.g., \code{"float32"}).}
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
Reload a Keras model/layer that was saved via \code{export_savedmodel()}.
}
\section{Examples}{
\if{html}{\out{<div class="sourceCode r">}}\preformatted{model <- keras_model_sequential(input_shape = c(784)) |> layer_dense(10)
model |> export_savedmodel("path/to/artifact")
}\if{html}{\out{</div>}}

\if{html}{\out{<div class="sourceCode">}}\preformatted{## Saved artifact at 'path/to/artifact'. The following endpoints are available:
##
## * Endpoint 'serve'
##   args_0 (POSITIONAL_ONLY): TensorSpec(shape=(None, 784), dtype=tf.float32, name='keras_tensor')
## Output Type:
##   TensorSpec(shape=(None, 10), dtype=tf.float32, name=None)
## Captures:
##   131061000739280: TensorSpec(shape=(), dtype=tf.resource, name=None)
##   131061000735824: TensorSpec(shape=(), dtype=tf.resource, name=None)

}\if{html}{\out{</div>}}

\if{html}{\out{<div class="sourceCode r">}}\preformatted{reloaded_layer <- layer_tfsm(filepath = "path/to/artifact")
input <- random_normal(c(2, 784))
output <- reloaded_layer(input)
stopifnot(all.equal(as.array(output), as.array(model(input))))
}\if{html}{\out{</div>}}

The reloaded object can be used like a regular Keras layer, and supports
training/fine-tuning of its trainable weights. Note that the reloaded
object retains none of the internal structure or custom methods of the
original object -- it's a brand new layer created around the saved
function.

\strong{Limitations:}
\itemize{
\item Only call endpoints with a single \code{inputs} tensor argument
(which may optionally be a named list/list of tensors) are supported.
For endpoints with multiple separate input tensor arguments, consider
subclassing \code{layer_tfsm} and implementing a \code{call()} method with a
custom signature.
\item If you need training-time behavior to differ from inference-time behavior
(i.e. if you need the reloaded object to support a \code{training=TRUE} argument
in \verb{__call__()}), make sure that the training-time call function is
saved as a standalone endpoint in the artifact, and provide its name
to the \code{layer_tfsm} via the \code{call_training_endpoint} argument.
}
}

\seealso{
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

Other saving and loading functions: \cr
\code{\link{export_savedmodel.keras.src.models.model.Model}()} \cr
\code{\link{load_model}()} \cr
\code{\link{load_model_weights}()} \cr
\code{\link{register_keras_serializable}()} \cr
\code{\link{save_model}()} \cr
\code{\link{save_model_config}()} \cr
\code{\link{save_model_weights}()} \cr
\code{\link{with_custom_object_scope}()} \cr
}
\concept{layers}
\concept{saving and loading functions}
