# Package index

## Models

### Create Models

- [`keras_model_sequential()`](https://keras3.posit.co/dev/reference/keras_model_sequential.md)
  : Keras Model composed of a linear stack of layers

- [`keras_model()`](https://keras3.posit.co/dev/reference/keras_model.md)
  : Keras Model (Functional API)

- [`keras_input()`](https://keras3.posit.co/dev/reference/keras_input.md)
  : Create a Keras tensor (Functional API input).

- [`clone_model()`](https://keras3.posit.co/dev/reference/clone_model.md)
  :

  Clone a Functional or Sequential `Model` instance.

- [`Model()`](https://keras3.posit.co/dev/reference/Model.md) :

  Subclass the base Keras `Model` Class

### Train Models

- [`compile(`*`<keras.src.models.model.Model>`*`)`](https://keras3.posit.co/dev/reference/compile.keras.src.models.model.Model.md)
  : Configure a model for training.
- [`fit(`*`<keras.src.models.model.Model>`*`)`](https://keras3.posit.co/dev/reference/fit.keras.src.models.model.Model.md)
  : Train a model for a fixed number of epochs (dataset iterations).
- [`plot(`*`<keras_training_history>`*`)`](https://keras3.posit.co/dev/reference/plot.keras_training_history.md)
  : Plot training history
- [`predict(`*`<keras.src.models.model.Model>`*`)`](https://keras3.posit.co/dev/reference/predict.keras.src.models.model.Model.md)
  : Generates output predictions for the input samples.
- [`evaluate(`*`<keras.src.models.model.Model>`*`)`](https://keras3.posit.co/dev/reference/evaluate.keras.src.models.model.Model.md)
  : Evaluate a Keras Model
- [`train_on_batch()`](https://keras3.posit.co/dev/reference/train_on_batch.md)
  : Runs a single gradient update on a single batch of data.
- [`predict_on_batch()`](https://keras3.posit.co/dev/reference/predict_on_batch.md)
  : Returns predictions for a single batch of samples.
- [`test_on_batch()`](https://keras3.posit.co/dev/reference/test_on_batch.md)
  : Test the model on a single batch of samples.
- [`freeze_weights()`](https://keras3.posit.co/dev/reference/freeze_weights.md)
  [`unfreeze_weights()`](https://keras3.posit.co/dev/reference/freeze_weights.md)
  : Freeze and unfreeze weights

### Inspect and Modify Models

- [`summary(`*`<keras.src.models.model.Model>`*`)`](https://keras3.posit.co/dev/reference/summary.keras.src.models.model.Model.md)
  [`format(`*`<keras.src.models.model.Model>`*`)`](https://keras3.posit.co/dev/reference/summary.keras.src.models.model.Model.md)
  [`print(`*`<keras.src.models.model.Model>`*`)`](https://keras3.posit.co/dev/reference/summary.keras.src.models.model.Model.md)
  : Print a summary of a Keras Model
- [`plot(`*`<keras.src.models.model.Model>`*`)`](https://keras3.posit.co/dev/reference/plot.keras.src.models.model.Model.md)
  : Plot a Keras model
- [`get_config()`](https://keras3.posit.co/dev/reference/get_config.md)
  [`from_config()`](https://keras3.posit.co/dev/reference/get_config.md)
  : Layer/Model configuration
- [`get_weights()`](https://keras3.posit.co/dev/reference/get_weights.md)
  [`set_weights()`](https://keras3.posit.co/dev/reference/get_weights.md)
  : Layer/Model weights as R arrays
- [`get_layer()`](https://keras3.posit.co/dev/reference/get_layer.md) :
  Retrieves a layer based on either its name (unique) or index.
- [`count_params()`](https://keras3.posit.co/dev/reference/count_params.md)
  : Count the total number of scalars composing the weights.
- [`pop_layer()`](https://keras3.posit.co/dev/reference/pop_layer.md) :
  Remove the last layer in a Sequential model
- [`quantize_weights()`](https://keras3.posit.co/dev/reference/quantize_weights.md)
  : Quantize the weights of a model.
- [`get_state_tree()`](https://keras3.posit.co/dev/reference/get_state_tree.md)
  : Retrieves tree-like structure of model variables.
- [`set_state_tree()`](https://keras3.posit.co/dev/reference/set_state_tree.md)
  : Assigns values to variables of the model.

### Save and Load Models

- [`save_model()`](https://keras3.posit.co/dev/reference/save_model.md)
  :

  Saves a model as a `.keras` file.

- [`load_model()`](https://keras3.posit.co/dev/reference/load_model.md)
  :

  Loads a model saved via
  [`save_model()`](https://keras3.posit.co/dev/reference/save_model.md).

- [`save_model_weights()`](https://keras3.posit.co/dev/reference/save_model_weights.md)
  : Saves all weights to a single file or sharded files.

- [`load_model_weights()`](https://keras3.posit.co/dev/reference/load_model_weights.md)
  : Load the weights from a single file or sharded files.

- [`save_model_config()`](https://keras3.posit.co/dev/reference/save_model_config.md)
  [`load_model_config()`](https://keras3.posit.co/dev/reference/save_model_config.md)
  : Save and load model configuration as JSON

- [`export_savedmodel(`*`<keras.src.models.model.Model>`*`)`](https://keras3.posit.co/dev/reference/export_savedmodel.keras.src.models.model.Model.md)
  : Export the model as an artifact for inference.

- [`layer_tfsm()`](https://keras3.posit.co/dev/reference/layer_tfsm.md)
  :

  Reload a Keras model/layer that was saved via
  [`export_savedmodel()`](https://rdrr.io/pkg/tensorflow/man/export_savedmodel.html).

- [`register_keras_serializable()`](https://keras3.posit.co/dev/reference/register_keras_serializable.md)
  : Registers a custom object with the Keras serialization framework.

## Layers

### Core Layers

- [`layer_dense()`](https://keras3.posit.co/dev/reference/layer_dense.md)
  : Just your regular densely-connected NN layer.

- [`layer_einsum_dense()`](https://keras3.posit.co/dev/reference/layer_einsum_dense.md)
  :

  A layer that uses `einsum` as the backing computation.

- [`layer_embedding()`](https://keras3.posit.co/dev/reference/layer_embedding.md)
  : Turns nonnegative integers (indexes) into dense vectors of fixed
  size.

- [`layer_identity()`](https://keras3.posit.co/dev/reference/layer_identity.md)
  : Identity layer.

- [`layer_lambda()`](https://keras3.posit.co/dev/reference/layer_lambda.md)
  :

  Wraps arbitrary expressions as a `Layer` object.

- [`layer_masking()`](https://keras3.posit.co/dev/reference/layer_masking.md)
  : Masks a sequence by using a mask value to skip timesteps.

### Reshaping Layers

- [`layer_cropping_1d()`](https://keras3.posit.co/dev/reference/layer_cropping_1d.md)
  : Cropping layer for 1D input (e.g. temporal sequence).
- [`layer_cropping_2d()`](https://keras3.posit.co/dev/reference/layer_cropping_2d.md)
  : Cropping layer for 2D input (e.g. picture).
- [`layer_cropping_3d()`](https://keras3.posit.co/dev/reference/layer_cropping_3d.md)
  : Cropping layer for 3D data (e.g. spatial or spatio-temporal).
- [`layer_flatten()`](https://keras3.posit.co/dev/reference/layer_flatten.md)
  : Flattens the input. Does not affect the batch size.
- [`layer_permute()`](https://keras3.posit.co/dev/reference/layer_permute.md)
  : Permutes the dimensions of the input according to a given pattern.
- [`layer_repeat_vector()`](https://keras3.posit.co/dev/reference/layer_repeat_vector.md)
  : Repeats the input n times.
- [`layer_reshape()`](https://keras3.posit.co/dev/reference/layer_reshape.md)
  : Layer that reshapes inputs into the given shape.
- [`layer_upsampling_1d()`](https://keras3.posit.co/dev/reference/layer_upsampling_1d.md)
  : Upsampling layer for 1D inputs.
- [`layer_upsampling_2d()`](https://keras3.posit.co/dev/reference/layer_upsampling_2d.md)
  : Upsampling layer for 2D inputs.
- [`layer_upsampling_3d()`](https://keras3.posit.co/dev/reference/layer_upsampling_3d.md)
  : Upsampling layer for 3D inputs.
- [`layer_zero_padding_1d()`](https://keras3.posit.co/dev/reference/layer_zero_padding_1d.md)
  : Zero-padding layer for 1D input (e.g. temporal sequence).
- [`layer_zero_padding_2d()`](https://keras3.posit.co/dev/reference/layer_zero_padding_2d.md)
  : Zero-padding layer for 2D input (e.g. picture).
- [`layer_zero_padding_3d()`](https://keras3.posit.co/dev/reference/layer_zero_padding_3d.md)
  : Zero-padding layer for 3D data (spatial or spatio-temporal).

### Convolutional Layers

- [`layer_conv_1d()`](https://keras3.posit.co/dev/reference/layer_conv_1d.md)
  : 1D convolution layer (e.g. temporal convolution).
- [`layer_conv_1d_transpose()`](https://keras3.posit.co/dev/reference/layer_conv_1d_transpose.md)
  : 1D transposed convolution layer.
- [`layer_conv_2d()`](https://keras3.posit.co/dev/reference/layer_conv_2d.md)
  : 2D convolution layer.
- [`layer_conv_2d_transpose()`](https://keras3.posit.co/dev/reference/layer_conv_2d_transpose.md)
  : 2D transposed convolution layer.
- [`layer_conv_3d()`](https://keras3.posit.co/dev/reference/layer_conv_3d.md)
  : 3D convolution layer.
- [`layer_conv_3d_transpose()`](https://keras3.posit.co/dev/reference/layer_conv_3d_transpose.md)
  : 3D transposed convolution layer.
- [`layer_depthwise_conv_1d()`](https://keras3.posit.co/dev/reference/layer_depthwise_conv_1d.md)
  : 1D depthwise convolution layer.
- [`layer_depthwise_conv_2d()`](https://keras3.posit.co/dev/reference/layer_depthwise_conv_2d.md)
  : 2D depthwise convolution layer.
- [`layer_separable_conv_1d()`](https://keras3.posit.co/dev/reference/layer_separable_conv_1d.md)
  : 1D separable convolution layer.
- [`layer_separable_conv_2d()`](https://keras3.posit.co/dev/reference/layer_separable_conv_2d.md)
  : 2D separable convolution layer.

### Pooling Layers

- [`layer_average_pooling_1d()`](https://keras3.posit.co/dev/reference/layer_average_pooling_1d.md)
  : Average pooling for temporal data.
- [`layer_average_pooling_2d()`](https://keras3.posit.co/dev/reference/layer_average_pooling_2d.md)
  : Average pooling operation for 2D spatial data.
- [`layer_average_pooling_3d()`](https://keras3.posit.co/dev/reference/layer_average_pooling_3d.md)
  : Average pooling operation for 3D data (spatial or spatio-temporal).
- [`layer_global_average_pooling_1d()`](https://keras3.posit.co/dev/reference/layer_global_average_pooling_1d.md)
  : Global average pooling operation for temporal data.
- [`layer_global_average_pooling_2d()`](https://keras3.posit.co/dev/reference/layer_global_average_pooling_2d.md)
  : Global average pooling operation for 2D data.
- [`layer_global_average_pooling_3d()`](https://keras3.posit.co/dev/reference/layer_global_average_pooling_3d.md)
  : Global average pooling operation for 3D data.
- [`layer_global_max_pooling_1d()`](https://keras3.posit.co/dev/reference/layer_global_max_pooling_1d.md)
  : Global max pooling operation for temporal data.
- [`layer_global_max_pooling_2d()`](https://keras3.posit.co/dev/reference/layer_global_max_pooling_2d.md)
  : Global max pooling operation for 2D data.
- [`layer_global_max_pooling_3d()`](https://keras3.posit.co/dev/reference/layer_global_max_pooling_3d.md)
  : Global max pooling operation for 3D data.
- [`layer_max_pooling_1d()`](https://keras3.posit.co/dev/reference/layer_max_pooling_1d.md)
  : Max pooling operation for 1D temporal data.
- [`layer_max_pooling_2d()`](https://keras3.posit.co/dev/reference/layer_max_pooling_2d.md)
  : Max pooling operation for 2D spatial data.
- [`layer_max_pooling_3d()`](https://keras3.posit.co/dev/reference/layer_max_pooling_3d.md)
  : Max pooling operation for 3D data (spatial or spatio-temporal).

### Activation Layers

- [`layer_activation()`](https://keras3.posit.co/dev/reference/layer_activation.md)
  : Applies an activation function to an output.
- [`layer_activation_elu()`](https://keras3.posit.co/dev/reference/layer_activation_elu.md)
  : Applies an Exponential Linear Unit function to an output.
- [`layer_activation_leaky_relu()`](https://keras3.posit.co/dev/reference/layer_activation_leaky_relu.md)
  : Leaky version of a Rectified Linear Unit activation layer.
- [`layer_activation_parametric_relu()`](https://keras3.posit.co/dev/reference/layer_activation_parametric_relu.md)
  : Parametric Rectified Linear Unit activation layer.
- [`layer_activation_relu()`](https://keras3.posit.co/dev/reference/layer_activation_relu.md)
  : Rectified Linear Unit activation function layer.
- [`layer_activation_softmax()`](https://keras3.posit.co/dev/reference/layer_activation_softmax.md)
  : Softmax activation layer.

### Recurrent Layers

- [`layer_bidirectional()`](https://keras3.posit.co/dev/reference/layer_bidirectional.md)
  : Bidirectional wrapper for RNNs.
- [`layer_conv_lstm_1d()`](https://keras3.posit.co/dev/reference/layer_conv_lstm_1d.md)
  : 1D Convolutional LSTM.
- [`layer_conv_lstm_2d()`](https://keras3.posit.co/dev/reference/layer_conv_lstm_2d.md)
  : 2D Convolutional LSTM.
- [`layer_conv_lstm_3d()`](https://keras3.posit.co/dev/reference/layer_conv_lstm_3d.md)
  : 3D Convolutional LSTM.
- [`layer_gru()`](https://keras3.posit.co/dev/reference/layer_gru.md) :
  Gated Recurrent Unit - Cho et al. 2014.
- [`layer_lstm()`](https://keras3.posit.co/dev/reference/layer_lstm.md)
  : Long Short-Term Memory layer - Hochreiter 1997.
- [`layer_rnn()`](https://keras3.posit.co/dev/reference/layer_rnn.md) :
  Base class for recurrent layers
- [`layer_simple_rnn()`](https://keras3.posit.co/dev/reference/layer_simple_rnn.md)
  : Fully-connected RNN where the output is to be fed back as the new
  input.
- [`layer_time_distributed()`](https://keras3.posit.co/dev/reference/layer_time_distributed.md)
  : This wrapper allows to apply a layer to every temporal slice of an
  input.
- [`rnn_cell_gru()`](https://keras3.posit.co/dev/reference/rnn_cell_gru.md)
  : Cell class for the GRU layer.
- [`rnn_cell_lstm()`](https://keras3.posit.co/dev/reference/rnn_cell_lstm.md)
  : Cell class for the LSTM layer.
- [`rnn_cell_simple()`](https://keras3.posit.co/dev/reference/rnn_cell_simple.md)
  : Cell class for SimpleRNN.
- [`rnn_cells_stack()`](https://keras3.posit.co/dev/reference/rnn_cells_stack.md)
  : Wrapper allowing a stack of RNN cells to behave as a single cell.
- [`reset_state()`](https://keras3.posit.co/dev/reference/reset_state.md)
  : Reset the state for a model, layer or metric.

### Attention Layers

- [`layer_additive_attention()`](https://keras3.posit.co/dev/reference/layer_additive_attention.md)
  : Additive attention layer, a.k.a. Bahdanau-style attention.
- [`layer_attention()`](https://keras3.posit.co/dev/reference/layer_attention.md)
  : Dot-product attention layer, a.k.a. Luong-style attention.
- [`layer_group_query_attention()`](https://keras3.posit.co/dev/reference/layer_group_query_attention.md)
  : Grouped Query Attention layer.
- [`layer_multi_head_attention()`](https://keras3.posit.co/dev/reference/layer_multi_head_attention.md)
  : Multi Head Attention layer.

### Normalization Layers

- [`layer_batch_normalization()`](https://keras3.posit.co/dev/reference/layer_batch_normalization.md)
  : Layer that normalizes its inputs.
- [`layer_group_normalization()`](https://keras3.posit.co/dev/reference/layer_group_normalization.md)
  : Group normalization layer.
- [`layer_layer_normalization()`](https://keras3.posit.co/dev/reference/layer_layer_normalization.md)
  : Layer normalization layer (Ba et al., 2016).
- [`layer_rms_normalization()`](https://keras3.posit.co/dev/reference/layer_rms_normalization.md)
  : Root Mean Square (RMS) Normalization layer.
- [`layer_spectral_normalization()`](https://keras3.posit.co/dev/reference/layer_spectral_normalization.md)
  : Performs spectral normalization on the weights of a target layer.
- [`layer_unit_normalization()`](https://keras3.posit.co/dev/reference/layer_unit_normalization.md)
  : Unit normalization layer.

### Regularization Layers

- [`layer_activity_regularization()`](https://keras3.posit.co/dev/reference/layer_activity_regularization.md)
  : Layer that applies an update to the cost function based input
  activity.
- [`layer_alpha_dropout()`](https://keras3.posit.co/dev/reference/layer_alpha_dropout.md)
  : Applies Alpha Dropout to the input.
- [`layer_dropout()`](https://keras3.posit.co/dev/reference/layer_dropout.md)
  : Applies dropout to the input.
- [`layer_gaussian_dropout()`](https://keras3.posit.co/dev/reference/layer_gaussian_dropout.md)
  : Apply multiplicative 1-centered Gaussian noise.
- [`layer_gaussian_noise()`](https://keras3.posit.co/dev/reference/layer_gaussian_noise.md)
  : Apply additive zero-centered Gaussian noise.
- [`layer_spatial_dropout_1d()`](https://keras3.posit.co/dev/reference/layer_spatial_dropout_1d.md)
  : Spatial 1D version of Dropout.
- [`layer_spatial_dropout_2d()`](https://keras3.posit.co/dev/reference/layer_spatial_dropout_2d.md)
  : Spatial 2D version of Dropout.
- [`layer_spatial_dropout_3d()`](https://keras3.posit.co/dev/reference/layer_spatial_dropout_3d.md)
  : Spatial 3D version of Dropout.

### Merging Layers

- [`layer_add()`](https://keras3.posit.co/dev/reference/layer_add.md) :
  Performs elementwise addition operation.
- [`layer_average()`](https://keras3.posit.co/dev/reference/layer_average.md)
  : Averages a list of inputs element-wise..
- [`layer_concatenate()`](https://keras3.posit.co/dev/reference/layer_concatenate.md)
  : Concatenates a list of inputs.
- [`layer_dot()`](https://keras3.posit.co/dev/reference/layer_dot.md) :
  Computes element-wise dot product of two tensors.
- [`layer_maximum()`](https://keras3.posit.co/dev/reference/layer_maximum.md)
  : Computes element-wise maximum on a list of inputs.
- [`layer_minimum()`](https://keras3.posit.co/dev/reference/layer_minimum.md)
  : Computes elementwise minimum on a list of inputs.
- [`layer_multiply()`](https://keras3.posit.co/dev/reference/layer_multiply.md)
  : Performs elementwise multiplication.
- [`layer_subtract()`](https://keras3.posit.co/dev/reference/layer_subtract.md)
  : Performs elementwise subtraction.

### Preprocessing Layers

- [`layer_aug_mix()`](https://keras3.posit.co/dev/reference/layer_aug_mix.md)
  : Performs the AugMix data augmentation technique.

- [`layer_auto_contrast()`](https://keras3.posit.co/dev/reference/layer_auto_contrast.md)
  : Performs the auto-contrast operation on an image.

- [`layer_category_encoding()`](https://keras3.posit.co/dev/reference/layer_category_encoding.md)
  : A preprocessing layer which encodes integer features.

- [`layer_center_crop()`](https://keras3.posit.co/dev/reference/layer_center_crop.md)
  : A preprocessing layer which crops images.

- [`layer_cut_mix()`](https://keras3.posit.co/dev/reference/layer_cut_mix.md)
  : CutMix data augmentation technique.

- [`layer_discretization()`](https://keras3.posit.co/dev/reference/layer_discretization.md)
  : A preprocessing layer which buckets continuous features by ranges.

- [`layer_equalization()`](https://keras3.posit.co/dev/reference/layer_equalization.md)
  : Preprocessing layer for histogram equalization on image channels.

- [`layer_feature_space()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_cross()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_custom()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_float()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_float_rescaled()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_float_normalized()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_float_discretized()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_integer_categorical()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_string_categorical()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_string_hashed()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_integer_hashed()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  : One-stop utility for preprocessing and encoding structured data.

- [`layer_hashed_crossing()`](https://keras3.posit.co/dev/reference/layer_hashed_crossing.md)
  : A preprocessing layer which crosses features using the "hashing
  trick".

- [`layer_hashing()`](https://keras3.posit.co/dev/reference/layer_hashing.md)
  : A preprocessing layer which hashes and bins categorical features.

- [`layer_integer_lookup()`](https://keras3.posit.co/dev/reference/layer_integer_lookup.md)
  : A preprocessing layer that maps integers to (possibly encoded)
  indices.

- [`layer_max_num_bounding_boxes()`](https://keras3.posit.co/dev/reference/layer_max_num_bounding_boxes.md)
  : Ensure the maximum number of bounding boxes.

- [`layer_mel_spectrogram()`](https://keras3.posit.co/dev/reference/layer_mel_spectrogram.md)
  : A preprocessing layer to convert raw audio signals to Mel
  spectrograms.

- [`layer_mix_up()`](https://keras3.posit.co/dev/reference/layer_mix_up.md)
  : MixUp implements the MixUp data augmentation technique.

- [`layer_normalization()`](https://keras3.posit.co/dev/reference/layer_normalization.md)
  : A preprocessing layer that normalizes continuous features.

- [`layer_rand_augment()`](https://keras3.posit.co/dev/reference/layer_rand_augment.md)
  : RandAugment performs the Rand Augment operation on input images.

- [`layer_random_brightness()`](https://keras3.posit.co/dev/reference/layer_random_brightness.md)
  : A preprocessing layer which randomly adjusts brightness during
  training.

- [`layer_random_color_degeneration()`](https://keras3.posit.co/dev/reference/layer_random_color_degeneration.md)
  : Randomly performs the color degeneration operation on given images.

- [`layer_random_color_jitter()`](https://keras3.posit.co/dev/reference/layer_random_color_jitter.md)
  : Randomly apply brightness, contrast, saturation

- [`layer_random_contrast()`](https://keras3.posit.co/dev/reference/layer_random_contrast.md)
  : A preprocessing layer which randomly adjusts contrast during
  training.

- [`layer_random_crop()`](https://keras3.posit.co/dev/reference/layer_random_crop.md)
  : A preprocessing layer which randomly crops images during training.

- [`layer_random_elastic_transform()`](https://keras3.posit.co/dev/reference/layer_random_elastic_transform.md)
  : A preprocessing layer that applies random elastic transformations.

- [`layer_random_erasing()`](https://keras3.posit.co/dev/reference/layer_random_erasing.md)
  : Random Erasing data augmentation technique.

- [`layer_random_flip()`](https://keras3.posit.co/dev/reference/layer_random_flip.md)
  : A preprocessing layer which randomly flips images during training.

- [`layer_random_gaussian_blur()`](https://keras3.posit.co/dev/reference/layer_random_gaussian_blur.md)
  : Applies random Gaussian blur to images for data augmentation.

- [`layer_random_grayscale()`](https://keras3.posit.co/dev/reference/layer_random_grayscale.md)
  : Preprocessing layer for random conversion of RGB images to
  grayscale.

- [`layer_random_hue()`](https://keras3.posit.co/dev/reference/layer_random_hue.md)
  : Randomly adjusts the hue on given images.

- [`layer_random_invert()`](https://keras3.posit.co/dev/reference/layer_random_invert.md)
  : Preprocessing layer for random inversion of image colors.

- [`layer_random_perspective()`](https://keras3.posit.co/dev/reference/layer_random_perspective.md)
  : A preprocessing layer that applies random perspective
  transformations.

- [`layer_random_posterization()`](https://keras3.posit.co/dev/reference/layer_random_posterization.md)
  : Reduces the number of bits for each color channel.

- [`layer_random_rotation()`](https://keras3.posit.co/dev/reference/layer_random_rotation.md)
  : A preprocessing layer which randomly rotates images during training.

- [`layer_random_saturation()`](https://keras3.posit.co/dev/reference/layer_random_saturation.md)
  : Randomly adjusts the saturation on given images.

- [`layer_random_sharpness()`](https://keras3.posit.co/dev/reference/layer_random_sharpness.md)
  : Randomly performs the sharpness operation on given images.

- [`layer_random_shear()`](https://keras3.posit.co/dev/reference/layer_random_shear.md)
  : A preprocessing layer that randomly applies shear transformations

- [`layer_random_translation()`](https://keras3.posit.co/dev/reference/layer_random_translation.md)
  : A preprocessing layer which randomly translates images during
  training.

- [`layer_random_zoom()`](https://keras3.posit.co/dev/reference/layer_random_zoom.md)
  : A preprocessing layer which randomly zooms images during training.

- [`layer_rescaling()`](https://keras3.posit.co/dev/reference/layer_rescaling.md)
  : A preprocessing layer which rescales input values to a new range.

- [`layer_resizing()`](https://keras3.posit.co/dev/reference/layer_resizing.md)
  : A preprocessing layer which resizes images.

- [`layer_solarization()`](https://keras3.posit.co/dev/reference/layer_solarization.md)
  :

  Applies `(max_value - pixel + min_value)` for each pixel in the image.

- [`layer_stft_spectrogram()`](https://keras3.posit.co/dev/reference/layer_stft_spectrogram.md)
  : Layer to compute the Short-Time Fourier Transform (STFT) on a 1D
  signal.

- [`layer_string_lookup()`](https://keras3.posit.co/dev/reference/layer_string_lookup.md)
  : A preprocessing layer that maps strings to (possibly encoded)
  indices.

- [`layer_text_vectorization()`](https://keras3.posit.co/dev/reference/layer_text_vectorization.md)
  [`get_vocabulary()`](https://keras3.posit.co/dev/reference/layer_text_vectorization.md)
  [`set_vocabulary()`](https://keras3.posit.co/dev/reference/layer_text_vectorization.md)
  : A preprocessing layer which maps text features to integer sequences.

- [`layer_pipeline()`](https://keras3.posit.co/dev/reference/layer_pipeline.md)
  : Applies a series of layers to an input.

- [`adapt()`](https://keras3.posit.co/dev/reference/adapt.md) : Fits the
  state of the preprocessing layer to the data being passed

### Compatability Layers

- [`layer_tfsm()`](https://keras3.posit.co/dev/reference/layer_tfsm.md)
  :

  Reload a Keras model/layer that was saved via
  [`export_savedmodel()`](https://rdrr.io/pkg/tensorflow/man/export_savedmodel.html).

- [`layer_jax_model_wrapper()`](https://keras3.posit.co/dev/reference/layer_jax_model_wrapper.md)
  : Keras Layer that wraps a JAX model.

- [`layer_flax_module_wrapper()`](https://keras3.posit.co/dev/reference/layer_flax_module_wrapper.md)
  :

  Keras Layer that wraps a [Flax](https://flax.readthedocs.io) module.

- [`layer_torch_module_wrapper()`](https://keras3.posit.co/dev/reference/layer_torch_module_wrapper.md)
  : Torch module wrapper layer.

### Custom Layers

- [`layer_lambda()`](https://keras3.posit.co/dev/reference/layer_lambda.md)
  :

  Wraps arbitrary expressions as a `Layer` object.

- [`Layer()`](https://keras3.posit.co/dev/reference/Layer.md) :

  Define a custom `Layer` class.

- [`keras_variable()`](https://keras3.posit.co/dev/reference/keras_variable.md)
  : Represents a backend-agnostic variable in Keras.

### Layer Methods

- [`get_config()`](https://keras3.posit.co/dev/reference/get_config.md)
  [`from_config()`](https://keras3.posit.co/dev/reference/get_config.md)
  : Layer/Model configuration
- [`get_weights()`](https://keras3.posit.co/dev/reference/get_weights.md)
  [`set_weights()`](https://keras3.posit.co/dev/reference/get_weights.md)
  : Layer/Model weights as R arrays
- [`count_params()`](https://keras3.posit.co/dev/reference/count_params.md)
  : Count the total number of scalars composing the weights.
- [`reset_state()`](https://keras3.posit.co/dev/reference/reset_state.md)
  : Reset the state for a model, layer or metric.

## Callbacks

- [`callback_model_checkpoint()`](https://keras3.posit.co/dev/reference/callback_model_checkpoint.md)
  : Callback to save the Keras model or model weights at some frequency.

- [`callback_backup_and_restore()`](https://keras3.posit.co/dev/reference/callback_backup_and_restore.md)
  : Callback to back up and restore the training state.

- [`callback_early_stopping()`](https://keras3.posit.co/dev/reference/callback_early_stopping.md)
  : Stop training when a monitored metric has stopped improving.

- [`callback_terminate_on_nan()`](https://keras3.posit.co/dev/reference/callback_terminate_on_nan.md)
  : Callback that terminates training when a NaN loss is encountered.

- [`callback_learning_rate_scheduler()`](https://keras3.posit.co/dev/reference/callback_learning_rate_scheduler.md)
  : Learning rate scheduler.

- [`callback_reduce_lr_on_plateau()`](https://keras3.posit.co/dev/reference/callback_reduce_lr_on_plateau.md)
  : Reduce learning rate when a metric has stopped improving.

- [`callback_csv_logger()`](https://keras3.posit.co/dev/reference/callback_csv_logger.md)
  : Callback that streams epoch results to a CSV file.

- [`callback_tensorboard()`](https://keras3.posit.co/dev/reference/callback_tensorboard.md)
  : Enable visualizations for TensorBoard.

- [`callback_remote_monitor()`](https://keras3.posit.co/dev/reference/callback_remote_monitor.md)
  : Callback used to stream events to a server.

- [`callback_lambda()`](https://keras3.posit.co/dev/reference/callback_lambda.md)
  : Callback for creating simple, custom callbacks on-the-fly.

- [`callback_swap_ema_weights()`](https://keras3.posit.co/dev/reference/callback_swap_ema_weights.md)
  : Swaps model weights and EMA weights before and after evaluation.

- [`Callback()`](https://keras3.posit.co/dev/reference/Callback.md) :

  Define a custom `Callback` class

## Operations

Functions that are safe to call with both symbolic and eager tensor.

### Core Operations

- [`op_associative_scan()`](https://keras3.posit.co/dev/reference/op_associative_scan.md)
  : Performs a scan with an associative binary operation, in parallel.

- [`op_cast()`](https://keras3.posit.co/dev/reference/op_cast.md) : Cast
  a tensor to the desired dtype.

- [`op_cond()`](https://keras3.posit.co/dev/reference/op_cond.md) :

  Conditionally applies `true_fn` or `false_fn`.

- [`op_convert_to_numpy()`](https://keras3.posit.co/dev/reference/op_convert_to_numpy.md)
  [`op_convert_to_array()`](https://keras3.posit.co/dev/reference/op_convert_to_numpy.md)
  : Convert a tensor to an R or NumPy array.

- [`op_convert_to_tensor()`](https://keras3.posit.co/dev/reference/op_convert_to_tensor.md)
  : Convert an array to a tensor.

- [`op_custom_gradient()`](https://keras3.posit.co/dev/reference/op_custom_gradient.md)
  : Decorator to define a function with a custom gradient.

- [`op_dtype()`](https://keras3.posit.co/dev/reference/op_dtype.md) :
  Return the dtype of the tensor input as a standardized string.

- [`op_fori_loop()`](https://keras3.posit.co/dev/reference/op_fori_loop.md)
  : For loop implementation.

- [`op_is_tensor()`](https://keras3.posit.co/dev/reference/op_is_tensor.md)
  : Check whether the given object is a tensor.

- [`op_map()`](https://keras3.posit.co/dev/reference/op_map.md) : Map a
  function over leading array axes.

- [`op_rearrange()`](https://keras3.posit.co/dev/reference/op_rearrange.md)
  : Rearranges the axes of a Keras tensor according to a specified
  pattern,

- [`op_scan()`](https://keras3.posit.co/dev/reference/op_scan.md) : Scan
  a function over leading array axes while carrying along state.

- [`op_scatter()`](https://keras3.posit.co/dev/reference/op_scatter.md)
  :

  Returns a tensor of shape `shape` where `indices` are set to `values`.

- [`op_scatter_update()`](https://keras3.posit.co/dev/reference/op_scatter_update.md)
  : Update inputs via updates at scattered (sparse) indices.

- [`op_searchsorted()`](https://keras3.posit.co/dev/reference/op_searchsorted.md)
  : Perform a binary search.

- [`op_shape()`](https://keras3.posit.co/dev/reference/op_shape.md) :
  Gets the shape of the tensor input.

- [`op_slice()`](https://keras3.posit.co/dev/reference/op_slice.md) :
  Return a slice of an input tensor.

- [`op_slice_update()`](https://keras3.posit.co/dev/reference/op_slice_update.md)
  : Update an input by slicing in a tensor of updated values.

- [`op_stop_gradient()`](https://keras3.posit.co/dev/reference/op_stop_gradient.md)
  : Stops gradient computation.

- [`op_subset()`](https://keras3.posit.co/dev/reference/op_subset.md)
  [`` `op_subset<-`() ``](https://keras3.posit.co/dev/reference/op_subset.md)
  [`op_subset_set()`](https://keras3.posit.co/dev/reference/op_subset.md)
  : Subset elements from a tensor

- [`op_switch()`](https://keras3.posit.co/dev/reference/op_switch.md) :

  Apply exactly one of the `branches` given by `index`.

- [`op_unstack()`](https://keras3.posit.co/dev/reference/op_unstack.md)
  : Unpacks the given dimension of a rank-R tensor into rank-(R-1)
  tensors.

- [`op_vectorized_map()`](https://keras3.posit.co/dev/reference/op_vectorized_map.md)
  :

  Parallel map of function `f` on the first axis of tensor(s)
  `elements`.

- [`op_while_loop()`](https://keras3.posit.co/dev/reference/op_while_loop.md)
  : While loop implementation.

### Math Operations

- [`op_erf()`](https://keras3.posit.co/dev/reference/op_erf.md) :

  Computes the error function of `x`, element-wise.

- [`op_erfinv()`](https://keras3.posit.co/dev/reference/op_erfinv.md) :

  Computes the inverse error function of `x`, element-wise.

- [`op_extract_sequences()`](https://keras3.posit.co/dev/reference/op_extract_sequences.md)
  :

  Expands the dimension of last axis into sequences of
  `sequence_length`.

- [`op_fft()`](https://keras3.posit.co/dev/reference/op_fft.md) :
  Computes the Fast Fourier Transform along last axis of input.

- [`op_fft2()`](https://keras3.posit.co/dev/reference/op_fft2.md) :
  Computes the 2D Fast Fourier Transform along the last two axes of
  input.

- [`op_ifft2()`](https://keras3.posit.co/dev/reference/op_ifft2.md) :
  Computes the 2D Inverse Fast Fourier Transform along the last two axes
  of

- [`op_in_top_k()`](https://keras3.posit.co/dev/reference/op_in_top_k.md)
  : Checks if the targets are in the top-k predictions.

- [`op_irfft()`](https://keras3.posit.co/dev/reference/op_irfft.md) :
  Inverse real-valued Fast Fourier transform along the last axis.

- [`op_istft()`](https://keras3.posit.co/dev/reference/op_istft.md) :
  Inverse Short-Time Fourier Transform along the last axis of the input.

- [`op_logsumexp()`](https://keras3.posit.co/dev/reference/op_logsumexp.md)
  : Computes the logarithm of sum of exponentials of elements in a
  tensor.

- [`op_qr()`](https://keras3.posit.co/dev/reference/op_qr.md) : Computes
  the QR decomposition of a tensor.

- [`op_rfft()`](https://keras3.posit.co/dev/reference/op_rfft.md) :
  Real-valued Fast Fourier Transform along the last axis of the input.

- [`op_rsqrt()`](https://keras3.posit.co/dev/reference/op_rsqrt.md) :
  Computes reciprocal of square root of x element-wise.

- [`op_segment_max()`](https://keras3.posit.co/dev/reference/op_segment_max.md)
  : Computes the max of segments in a tensor.

- [`op_segment_sum()`](https://keras3.posit.co/dev/reference/op_segment_sum.md)
  : Computes the sum of segments in a tensor.

- [`op_solve()`](https://keras3.posit.co/dev/reference/op_solve.md) :

  Solves a linear system of equations given by `a x = b`.

- [`op_stft()`](https://keras3.posit.co/dev/reference/op_stft.md) :
  Short-Time Fourier Transform along the last axis of the input.

- [`op_top_k()`](https://keras3.posit.co/dev/reference/op_top_k.md) :
  Finds the top-k values and their indices in a tensor.

### General Tensor Operations

- [`op_abs()`](https://keras3.posit.co/dev/reference/op_abs.md) :
  Compute the absolute value element-wise.

- [`op_add()`](https://keras3.posit.co/dev/reference/op_add.md) : Add
  arguments element-wise.

- [`op_all()`](https://keras3.posit.co/dev/reference/op_all.md) :

  Test whether all array elements along a given axis evaluate to `TRUE`.

- [`op_angle()`](https://keras3.posit.co/dev/reference/op_angle.md) :
  Element-wise angle of a complex tensor.

- [`op_any()`](https://keras3.posit.co/dev/reference/op_any.md) :

  Test whether any array element along a given axis evaluates to `TRUE`.

- [`op_append()`](https://keras3.posit.co/dev/reference/op_append.md) :

  Append tensor `x2` to the end of tensor `x1`.

- [`op_arange()`](https://keras3.posit.co/dev/reference/op_arange.md) :
  Return evenly spaced values within a given interval.

- [`op_arccos()`](https://keras3.posit.co/dev/reference/op_arccos.md) :
  Trigonometric inverse cosine, element-wise.

- [`op_arccosh()`](https://keras3.posit.co/dev/reference/op_arccosh.md)
  : Inverse hyperbolic cosine, element-wise.

- [`op_arcsin()`](https://keras3.posit.co/dev/reference/op_arcsin.md) :
  Inverse sine, element-wise.

- [`op_arcsinh()`](https://keras3.posit.co/dev/reference/op_arcsinh.md)
  : Inverse hyperbolic sine, element-wise.

- [`op_arctan()`](https://keras3.posit.co/dev/reference/op_arctan.md) :
  Trigonometric inverse tangent, element-wise.

- [`op_arctan2()`](https://keras3.posit.co/dev/reference/op_arctan2.md)
  :

  Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.

- [`op_arctanh()`](https://keras3.posit.co/dev/reference/op_arctanh.md)
  : Inverse hyperbolic tangent, element-wise.

- [`op_argmax()`](https://keras3.posit.co/dev/reference/op_argmax.md) :
  Returns the indices of the maximum values along an axis.

- [`op_argmin()`](https://keras3.posit.co/dev/reference/op_argmin.md) :
  Returns the indices of the minimum values along an axis.

- [`op_argpartition()`](https://keras3.posit.co/dev/reference/op_argpartition.md)
  : Performs an indirect partition along the given axis.

- [`op_argsort()`](https://keras3.posit.co/dev/reference/op_argsort.md)
  : Returns the indices that would sort a tensor.

- [`op_array()`](https://keras3.posit.co/dev/reference/op_array.md) :
  Create a tensor.

- [`op_average()`](https://keras3.posit.co/dev/reference/op_average.md)
  : Compute the weighted average along the specified axis.

- [`op_bartlett()`](https://keras3.posit.co/dev/reference/op_bartlett.md)
  : Bartlett window function.

- [`op_bincount()`](https://keras3.posit.co/dev/reference/op_bincount.md)
  : Count the number of occurrences of each value in a tensor of
  integers.

- [`op_bitwise_and()`](https://keras3.posit.co/dev/reference/op_bitwise_and.md)
  : Compute the bit-wise AND of two arrays element-wise.

- [`op_bitwise_invert()`](https://keras3.posit.co/dev/reference/op_bitwise_invert.md)
  : Compute bit-wise inversion, or bit-wise NOT, element-wise.

- [`op_bitwise_left_shift()`](https://keras3.posit.co/dev/reference/op_bitwise_left_shift.md)
  : Shift the bits of an integer to the left.

- [`op_bitwise_not()`](https://keras3.posit.co/dev/reference/op_bitwise_not.md)
  : Compute bit-wise inversion, or bit-wise NOT, element-wise.

- [`op_bitwise_or()`](https://keras3.posit.co/dev/reference/op_bitwise_or.md)
  : Compute the bit-wise OR of two arrays element-wise.

- [`op_bitwise_right_shift()`](https://keras3.posit.co/dev/reference/op_bitwise_right_shift.md)
  : Shift the bits of an integer to the right.

- [`op_bitwise_xor()`](https://keras3.posit.co/dev/reference/op_bitwise_xor.md)
  : Compute the bit-wise XOR of two arrays element-wise.

- [`op_blackman()`](https://keras3.posit.co/dev/reference/op_blackman.md)
  : Blackman window function.

- [`op_broadcast_to()`](https://keras3.posit.co/dev/reference/op_broadcast_to.md)
  : Broadcast a tensor to a new shape.

- [`op_cbrt()`](https://keras3.posit.co/dev/reference/op_cbrt.md) :
  Computes the cube root of the input tensor, element-wise.

- [`op_ceil()`](https://keras3.posit.co/dev/reference/op_ceil.md) :
  Return the ceiling of the input, element-wise.

- [`op_clip()`](https://keras3.posit.co/dev/reference/op_clip.md) : Clip
  (limit) the values in a tensor.

- [`op_concatenate()`](https://keras3.posit.co/dev/reference/op_concatenate.md)
  : Join a sequence of tensors along an existing axis.

- [`op_conj()`](https://keras3.posit.co/dev/reference/op_conj.md) :
  Returns the complex conjugate, element-wise.

- [`op_copy()`](https://keras3.posit.co/dev/reference/op_copy.md) :

  Returns a copy of `x`.

- [`op_corrcoef()`](https://keras3.posit.co/dev/reference/op_corrcoef.md)
  : Compute the Pearson correlation coefficient matrix.

- [`op_correlate()`](https://keras3.posit.co/dev/reference/op_correlate.md)
  : Compute the cross-correlation of two 1-dimensional tensors.

- [`op_cos()`](https://keras3.posit.co/dev/reference/op_cos.md) :
  Cosine, element-wise.

- [`op_cosh()`](https://keras3.posit.co/dev/reference/op_cosh.md) :
  Hyperbolic cosine, element-wise.

- [`op_count_nonzero()`](https://keras3.posit.co/dev/reference/op_count_nonzero.md)
  :

  Counts the number of non-zero values in `x` along the given `axis`.

- [`op_cross()`](https://keras3.posit.co/dev/reference/op_cross.md) :
  Returns the cross product of two (arrays of) vectors.

- [`op_ctc_decode()`](https://keras3.posit.co/dev/reference/op_ctc_decode.md)
  : Decodes the output of a CTC model.

- [`op_cumprod()`](https://keras3.posit.co/dev/reference/op_cumprod.md)
  : Return the cumulative product of elements along a given axis.

- [`op_cumsum()`](https://keras3.posit.co/dev/reference/op_cumsum.md) :
  Returns the cumulative sum of elements along a given axis.

- [`op_deg2rad()`](https://keras3.posit.co/dev/reference/op_deg2rad.md)
  : Convert angles from degrees to radians.

- [`op_diag()`](https://keras3.posit.co/dev/reference/op_diag.md) :
  Extract a diagonal or construct a diagonal array.

- [`op_diagflat()`](https://keras3.posit.co/dev/reference/op_diagflat.md)
  : Create a two-dimensional array with the flattened input diagonal.

- [`op_diagonal()`](https://keras3.posit.co/dev/reference/op_diagonal.md)
  : Return specified diagonals.

- [`op_diff()`](https://keras3.posit.co/dev/reference/op_diff.md) :
  Calculate the n-th discrete difference along the given axis.

- [`op_digitize()`](https://keras3.posit.co/dev/reference/op_digitize.md)
  :

  Returns the indices of the bins to which each value in `x` belongs.

- [`op_divide()`](https://keras3.posit.co/dev/reference/op_divide.md) :
  Divide arguments element-wise.

- [`op_divide_no_nan()`](https://keras3.posit.co/dev/reference/op_divide_no_nan.md)
  : Safe element-wise division which returns 0 where the denominator is
  0.

- [`op_dot()`](https://keras3.posit.co/dev/reference/op_dot.md) : Dot
  product of two tensors.

- [`op_einsum()`](https://keras3.posit.co/dev/reference/op_einsum.md) :
  Evaluates the Einstein summation convention on the operands.

- [`op_empty()`](https://keras3.posit.co/dev/reference/op_empty.md) :
  Return a tensor of given shape and type filled with uninitialized
  data.

- [`op_equal()`](https://keras3.posit.co/dev/reference/op_equal.md) :

  Returns `(x1 == x2)` element-wise.

- [`op_exp()`](https://keras3.posit.co/dev/reference/op_exp.md) :
  Calculate the exponential of all elements in the input tensor.

- [`op_exp2()`](https://keras3.posit.co/dev/reference/op_exp2.md) :
  Calculate the base-2 exponential of all elements in the input tensor.

- [`op_expand_dims()`](https://keras3.posit.co/dev/reference/op_expand_dims.md)
  : Expand the shape of a tensor.

- [`op_expm1()`](https://keras3.posit.co/dev/reference/op_expm1.md) :

  Calculate `exp(x) - 1` for all elements in the tensor.

- [`op_eye()`](https://keras3.posit.co/dev/reference/op_eye.md) : Return
  a 2-D tensor with ones on the diagonal and zeros elsewhere.

- [`op_flip()`](https://keras3.posit.co/dev/reference/op_flip.md) :
  Reverse the order of elements in the tensor along the given axis.

- [`op_floor()`](https://keras3.posit.co/dev/reference/op_floor.md) :
  Return the floor of the input, element-wise.

- [`op_floor_divide()`](https://keras3.posit.co/dev/reference/op_floor_divide.md)
  : Returns the largest integer smaller or equal to the division of
  inputs.

- [`op_full()`](https://keras3.posit.co/dev/reference/op_full.md) :

  Return a new tensor of given shape and type, filled with `fill_value`.

- [`op_full_like()`](https://keras3.posit.co/dev/reference/op_full_like.md)
  : Return a full tensor with the same shape and type as the given
  tensor.

- [`op_get_item()`](https://keras3.posit.co/dev/reference/op_get_item.md)
  :

  Return `x[key]`.

- [`op_greater()`](https://keras3.posit.co/dev/reference/op_greater.md)
  :

  Return the truth value of `x1 > x2` element-wise.

- [`op_greater_equal()`](https://keras3.posit.co/dev/reference/op_greater_equal.md)
  :

  Return the truth value of `x1 >= x2` element-wise.

- [`op_hamming()`](https://keras3.posit.co/dev/reference/op_hamming.md)
  : Hamming window function.

- [`op_hanning()`](https://keras3.posit.co/dev/reference/op_hanning.md)
  : Hanning window function.

- [`op_heaviside()`](https://keras3.posit.co/dev/reference/op_heaviside.md)
  : Heaviside step function.

- [`op_histogram()`](https://keras3.posit.co/dev/reference/op_histogram.md)
  :

  Computes a histogram of the data tensor `x`.

- [`op_hstack()`](https://keras3.posit.co/dev/reference/op_hstack.md) :
  Stack tensors in sequence horizontally (column wise).

- [`op_identity()`](https://keras3.posit.co/dev/reference/op_identity.md)
  : Return the identity tensor.

- [`op_imag()`](https://keras3.posit.co/dev/reference/op_imag.md) :
  Return the imaginary part of the complex argument.

- [`op_inner()`](https://keras3.posit.co/dev/reference/op_inner.md) :
  Return the inner product of two tensors.

- [`op_isclose()`](https://keras3.posit.co/dev/reference/op_isclose.md)
  : Return whether two tensors are element-wise almost equal.

- [`op_isfinite()`](https://keras3.posit.co/dev/reference/op_isfinite.md)
  : Return whether a tensor is finite, element-wise.

- [`op_isinf()`](https://keras3.posit.co/dev/reference/op_isinf.md) :
  Test element-wise for positive or negative infinity.

- [`op_isnan()`](https://keras3.posit.co/dev/reference/op_isnan.md) :
  Test element-wise for NaN and return result as a boolean tensor.

- [`op_kaiser()`](https://keras3.posit.co/dev/reference/op_kaiser.md) :
  Kaiser window function.

- [`op_left_shift()`](https://keras3.posit.co/dev/reference/op_left_shift.md)
  : Shift the bits of an integer to the left.

- [`op_less()`](https://keras3.posit.co/dev/reference/op_less.md) :

  Return the truth value of `x1 < x2` element-wise.

- [`op_less_equal()`](https://keras3.posit.co/dev/reference/op_less_equal.md)
  :

  Return the truth value of `x1 <= x2` element-wise.

- [`op_linspace()`](https://keras3.posit.co/dev/reference/op_linspace.md)
  : Return evenly spaced numbers over a specified interval.

- [`op_log()`](https://keras3.posit.co/dev/reference/op_log.md) :
  Natural logarithm, element-wise.

- [`op_log10()`](https://keras3.posit.co/dev/reference/op_log10.md) :
  Return the base 10 logarithm of the input tensor, element-wise.

- [`op_log1p()`](https://keras3.posit.co/dev/reference/op_log1p.md) :

  Returns the natural logarithm of one plus the `x`, element-wise.

- [`op_log2()`](https://keras3.posit.co/dev/reference/op_log2.md) :

  Base-2 logarithm of `x`, element-wise.

- [`op_logaddexp()`](https://keras3.posit.co/dev/reference/op_logaddexp.md)
  : Logarithm of the sum of exponentiations of the inputs.

- [`op_logdet()`](https://keras3.posit.co/dev/reference/op_logdet.md) :
  Computes log of the determinant of a hermitian positive definite
  matrix.

- [`op_logical_and()`](https://keras3.posit.co/dev/reference/op_logical_and.md)
  : Computes the element-wise logical AND of the given input tensors.

- [`op_logical_not()`](https://keras3.posit.co/dev/reference/op_logical_not.md)
  : Computes the element-wise NOT of the given input tensor.

- [`op_logical_or()`](https://keras3.posit.co/dev/reference/op_logical_or.md)
  : Computes the element-wise logical OR of the given input tensors.

- [`op_logical_xor()`](https://keras3.posit.co/dev/reference/op_logical_xor.md)
  :

  Compute the truth value of `x1 XOR x2`, element-wise.

- [`op_logspace()`](https://keras3.posit.co/dev/reference/op_logspace.md)
  : Returns numbers spaced evenly on a log scale.

- [`op_lstsq()`](https://keras3.posit.co/dev/reference/op_lstsq.md) :
  Return the least-squares solution to a linear matrix equation.

- [`op_matmul()`](https://keras3.posit.co/dev/reference/op_matmul.md) :
  Matrix product of two tensors.

- [`op_max()`](https://keras3.posit.co/dev/reference/op_max.md) : Return
  the maximum of a tensor or maximum along an axis.

- [`op_maximum()`](https://keras3.posit.co/dev/reference/op_maximum.md)
  [`op_pmax()`](https://keras3.posit.co/dev/reference/op_maximum.md) :

  Element-wise maximum of `x1` and `x2`.

- [`op_mean()`](https://keras3.posit.co/dev/reference/op_mean.md) :
  Compute the arithmetic mean along the specified axes.

- [`op_median()`](https://keras3.posit.co/dev/reference/op_median.md) :
  Compute the median along the specified axis.

- [`op_meshgrid()`](https://keras3.posit.co/dev/reference/op_meshgrid.md)
  : Creates grids of coordinates from coordinate vectors.

- [`op_min()`](https://keras3.posit.co/dev/reference/op_min.md) : Return
  the minimum of a tensor or minimum along an axis.

- [`op_minimum()`](https://keras3.posit.co/dev/reference/op_minimum.md)
  [`op_pmin()`](https://keras3.posit.co/dev/reference/op_minimum.md) :

  Element-wise minimum of `x1` and `x2`.

- [`op_mod()`](https://keras3.posit.co/dev/reference/op_mod.md) :
  Returns the element-wise remainder of division.

- [`op_moveaxis()`](https://keras3.posit.co/dev/reference/op_moveaxis.md)
  : Move axes of a tensor to new positions.

- [`op_multiply()`](https://keras3.posit.co/dev/reference/op_multiply.md)
  : Multiply arguments element-wise.

- [`op_nan_to_num()`](https://keras3.posit.co/dev/reference/op_nan_to_num.md)
  : Replace NaN with zero and infinity with large finite numbers.

- [`op_ndim()`](https://keras3.posit.co/dev/reference/op_ndim.md) :
  Return the number of dimensions of a tensor.

- [`op_negative()`](https://keras3.posit.co/dev/reference/op_negative.md)
  : Numerical negative, element-wise.

- [`op_nonzero()`](https://keras3.posit.co/dev/reference/op_nonzero.md)
  : Return the indices of the elements that are non-zero.

- [`op_not_equal()`](https://keras3.posit.co/dev/reference/op_not_equal.md)
  :

  Return `(x1 != x2)` element-wise.

- [`op_ones()`](https://keras3.posit.co/dev/reference/op_ones.md) :
  Return a new tensor of given shape and type, filled with ones.

- [`op_ones_like()`](https://keras3.posit.co/dev/reference/op_ones_like.md)
  :

  Return a tensor of ones with the same shape and type of `x`.

- [`op_outer()`](https://keras3.posit.co/dev/reference/op_outer.md) :
  Compute the outer product of two vectors.

- [`op_pad()`](https://keras3.posit.co/dev/reference/op_pad.md) : Pad a
  tensor.

- [`op_power()`](https://keras3.posit.co/dev/reference/op_power.md) :
  First tensor elements raised to powers from second tensor,
  element-wise.

- [`op_prod()`](https://keras3.posit.co/dev/reference/op_prod.md) :
  Return the product of tensor elements over a given axis.

- [`op_quantile()`](https://keras3.posit.co/dev/reference/op_quantile.md)
  : Compute the q-th quantile(s) of the data along the specified axis.

- [`op_ravel()`](https://keras3.posit.co/dev/reference/op_ravel.md) :
  Return a contiguous flattened tensor.

- [`op_real()`](https://keras3.posit.co/dev/reference/op_real.md) :
  Return the real part of the complex argument.

- [`op_reciprocal()`](https://keras3.posit.co/dev/reference/op_reciprocal.md)
  : Return the reciprocal of the argument, element-wise.

- [`op_repeat()`](https://keras3.posit.co/dev/reference/op_repeat.md) :
  Repeat each element of a tensor after themselves.

- [`op_reshape()`](https://keras3.posit.co/dev/reference/op_reshape.md)
  : Gives a new shape to a tensor without changing its data.

- [`op_right_shift()`](https://keras3.posit.co/dev/reference/op_right_shift.md)
  : Shift the bits of an integer to the right.

- [`op_roll()`](https://keras3.posit.co/dev/reference/op_roll.md) : Roll
  tensor elements along a given axis.

- [`op_rot90()`](https://keras3.posit.co/dev/reference/op_rot90.md) :
  Rotate an array by 90 degrees in the plane specified by axes.

- [`op_round()`](https://keras3.posit.co/dev/reference/op_round.md) :
  Evenly round to the given number of decimals.

- [`op_saturate_cast()`](https://keras3.posit.co/dev/reference/op_saturate_cast.md)
  : Performs a safe saturating cast to the desired dtype.

- [`op_select()`](https://keras3.posit.co/dev/reference/op_select.md) :

  Return elements from `choicelist`, based on conditions in `condlist`.

- [`op_sign()`](https://keras3.posit.co/dev/reference/op_sign.md) :

  Returns a tensor with the signs of the elements of `x`.

- [`op_signbit()`](https://keras3.posit.co/dev/reference/op_signbit.md)
  :

  Return the sign bit of the elements of `x`.

- [`op_sin()`](https://keras3.posit.co/dev/reference/op_sin.md) :
  Trigonometric sine, element-wise.

- [`op_sinh()`](https://keras3.posit.co/dev/reference/op_sinh.md) :
  Hyperbolic sine, element-wise.

- [`op_size()`](https://keras3.posit.co/dev/reference/op_size.md) :
  Return the number of elements in a tensor.

- [`op_sort()`](https://keras3.posit.co/dev/reference/op_sort.md) :

  Sorts the elements of `x` along a given axis in ascending order.

- [`op_split()`](https://keras3.posit.co/dev/reference/op_split.md) :
  Split a tensor into chunks.

- [`op_sqrt()`](https://keras3.posit.co/dev/reference/op_sqrt.md) :
  Return the non-negative square root of a tensor, element-wise.

- [`op_square()`](https://keras3.posit.co/dev/reference/op_square.md) :
  Return the element-wise square of the input.

- [`op_squeeze()`](https://keras3.posit.co/dev/reference/op_squeeze.md)
  :

  Remove axes of length one from `x`.

- [`op_stack()`](https://keras3.posit.co/dev/reference/op_stack.md) :
  Join a sequence of tensors along a new axis.

- [`op_std()`](https://keras3.posit.co/dev/reference/op_std.md) :
  Compute the standard deviation along the specified axis.

- [`op_subtract()`](https://keras3.posit.co/dev/reference/op_subtract.md)
  : Subtract arguments element-wise.

- [`op_sum()`](https://keras3.posit.co/dev/reference/op_sum.md) : Sum of
  a tensor over the given axes.

- [`op_swapaxes()`](https://keras3.posit.co/dev/reference/op_swapaxes.md)
  : Interchange two axes of a tensor.

- [`op_take()`](https://keras3.posit.co/dev/reference/op_take.md) : Take
  elements from a tensor along an axis.

- [`op_take_along_axis()`](https://keras3.posit.co/dev/reference/op_take_along_axis.md)
  :

  Select values from `x` at the 1-D `indices` along the given axis.

- [`op_tan()`](https://keras3.posit.co/dev/reference/op_tan.md) :
  Compute tangent, element-wise.

- [`op_tanh()`](https://keras3.posit.co/dev/reference/op_tanh.md) :
  Hyperbolic tangent, element-wise.

- [`op_tensordot()`](https://keras3.posit.co/dev/reference/op_tensordot.md)
  : Compute the tensor dot product along specified axes.

- [`op_tile()`](https://keras3.posit.co/dev/reference/op_tile.md) :

  Repeat `x` the number of times given by `repeats`.

- [`op_trace()`](https://keras3.posit.co/dev/reference/op_trace.md) :
  Return the sum along diagonals of the tensor.

- [`op_transpose()`](https://keras3.posit.co/dev/reference/op_transpose.md)
  :

  Returns a tensor with `axes` transposed.

- [`op_tri()`](https://keras3.posit.co/dev/reference/op_tri.md) : Return
  a tensor with ones at and below a diagonal and zeros elsewhere.

- [`op_tril()`](https://keras3.posit.co/dev/reference/op_tril.md) :
  Return lower triangle of a tensor.

- [`op_triu()`](https://keras3.posit.co/dev/reference/op_triu.md) :
  Return upper triangle of a tensor.

- [`op_trunc()`](https://keras3.posit.co/dev/reference/op_trunc.md) :
  Return the truncated value of the input, element-wise.

- [`op_var()`](https://keras3.posit.co/dev/reference/op_var.md) :
  Compute the variance along the specified axes.

- [`op_vdot()`](https://keras3.posit.co/dev/reference/op_vdot.md) :
  Return the dot product of two vectors.

- [`op_vectorize()`](https://keras3.posit.co/dev/reference/op_vectorize.md)
  : Turn a function into a vectorized function.

- [`op_view_as_complex()`](https://keras3.posit.co/dev/reference/op_view_as_complex.md)
  : Convert a real tensor with two channels into a complex tensor.

- [`op_view_as_real()`](https://keras3.posit.co/dev/reference/op_view_as_real.md)
  : Convert a complex tensor into a stacked real representation.

- [`op_vstack()`](https://keras3.posit.co/dev/reference/op_vstack.md) :
  Stack tensors in sequence vertically (row wise).

- [`op_where()`](https://keras3.posit.co/dev/reference/op_where.md) :

  Return elements chosen from `x1` or `x2` depending on `condition`.

- [`op_zeros()`](https://keras3.posit.co/dev/reference/op_zeros.md) :
  Return a new tensor of given shape and type, filled with zeros.

- [`op_zeros_like()`](https://keras3.posit.co/dev/reference/op_zeros_like.md)
  :

  Return a tensor of zeros with the same shape and type as `x`.

### Neural Network Operations

- [`op_average_pool()`](https://keras3.posit.co/dev/reference/op_average_pool.md)
  : Average pooling operation.

- [`op_batch_normalization()`](https://keras3.posit.co/dev/reference/op_batch_normalization.md)
  :

  Normalizes `x` by `mean` and `variance`.

- [`op_binary_crossentropy()`](https://keras3.posit.co/dev/reference/op_binary_crossentropy.md)
  : Computes binary cross-entropy loss between target and output tensor.

- [`op_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/op_categorical_crossentropy.md)
  : Computes categorical cross-entropy loss between target and output
  tensor.

- [`op_celu()`](https://keras3.posit.co/dev/reference/op_celu.md) :
  Continuously-differentiable exponential linear unit.

- [`op_conv()`](https://keras3.posit.co/dev/reference/op_conv.md) :
  General N-D convolution.

- [`op_conv_transpose()`](https://keras3.posit.co/dev/reference/op_conv_transpose.md)
  : General N-D convolution transpose.

- [`op_ctc_loss()`](https://keras3.posit.co/dev/reference/op_ctc_loss.md)
  : CTC (Connectionist Temporal Classification) loss.

- [`op_depthwise_conv()`](https://keras3.posit.co/dev/reference/op_depthwise_conv.md)
  : General N-D depthwise convolution.

- [`op_dot_product_attention()`](https://keras3.posit.co/dev/reference/op_dot_product_attention.md)
  : Scaled dot product attention function.

- [`op_elu()`](https://keras3.posit.co/dev/reference/op_elu.md) :
  Exponential Linear Unit activation function.

- [`op_gelu()`](https://keras3.posit.co/dev/reference/op_gelu.md) :
  Gaussian Error Linear Unit (GELU) activation function.

- [`op_glu()`](https://keras3.posit.co/dev/reference/op_glu.md) : Gated
  Linear Unit (GLU) activation function.

- [`op_hard_shrink()`](https://keras3.posit.co/dev/reference/op_hard_shrink.md)
  : Hard Shrink activation function.

- [`op_hard_sigmoid()`](https://keras3.posit.co/dev/reference/op_hard_sigmoid.md)
  : Hard sigmoid activation function.

- [`op_hard_silu()`](https://keras3.posit.co/dev/reference/op_hard_silu.md)
  [`op_hard_swish()`](https://keras3.posit.co/dev/reference/op_hard_silu.md)
  : Hard SiLU activation function, also known as Hard Swish.

- [`op_hard_tanh()`](https://keras3.posit.co/dev/reference/op_hard_tanh.md)
  : Applies the HardTanh function element-wise.

- [`op_layer_normalization()`](https://keras3.posit.co/dev/reference/op_layer_normalization.md)
  : Layer normalization (Ba et al., 2016).

- [`op_leaky_relu()`](https://keras3.posit.co/dev/reference/op_leaky_relu.md)
  : Leaky version of a Rectified Linear Unit activation function.

- [`op_log_sigmoid()`](https://keras3.posit.co/dev/reference/op_log_sigmoid.md)
  : Logarithm of the sigmoid activation function.

- [`op_log_softmax()`](https://keras3.posit.co/dev/reference/op_log_softmax.md)
  : Log-softmax activation function.

- [`op_max_pool()`](https://keras3.posit.co/dev/reference/op_max_pool.md)
  : Max pooling operation.

- [`op_moments()`](https://keras3.posit.co/dev/reference/op_moments.md)
  :

  Calculates the mean and variance of `x`.

- [`op_multi_hot()`](https://keras3.posit.co/dev/reference/op_multi_hot.md)
  : Encodes integer labels as multi-hot vectors.

- [`op_normalize()`](https://keras3.posit.co/dev/reference/op_normalize.md)
  :

  Normalizes `x` over the specified axis.

- [`op_one_hot()`](https://keras3.posit.co/dev/reference/op_one_hot.md)
  :

  Converts integer tensor `x` into a one-hot tensor.

- [`op_polar()`](https://keras3.posit.co/dev/reference/op_polar.md) :
  Constructs a complex tensor whose elements are Cartesian

- [`op_psnr()`](https://keras3.posit.co/dev/reference/op_psnr.md) : Peak
  Signal-to-Noise Ratio (PSNR) function.

- [`op_relu()`](https://keras3.posit.co/dev/reference/op_relu.md) :
  Rectified linear unit activation function.

- [`op_relu6()`](https://keras3.posit.co/dev/reference/op_relu6.md) :
  Rectified linear unit activation function with upper bound of 6.

- [`op_rms_normalization()`](https://keras3.posit.co/dev/reference/op_rms_normalization.md)
  :

  Performs Root Mean Square (RMS) normalization on `x`.

- [`op_selu()`](https://keras3.posit.co/dev/reference/op_selu.md) :
  Scaled Exponential Linear Unit (SELU) activation function.

- [`op_separable_conv()`](https://keras3.posit.co/dev/reference/op_separable_conv.md)
  : General N-D separable convolution.

- [`op_sigmoid()`](https://keras3.posit.co/dev/reference/op_sigmoid.md)
  : Sigmoid activation function.

- [`op_silu()`](https://keras3.posit.co/dev/reference/op_silu.md) :
  Sigmoid Linear Unit (SiLU) activation function, also known as Swish.

- [`op_soft_shrink()`](https://keras3.posit.co/dev/reference/op_soft_shrink.md)
  : Soft Shrink activation function.

- [`op_softmax()`](https://keras3.posit.co/dev/reference/op_softmax.md)
  : Softmax activation function.

- [`op_softplus()`](https://keras3.posit.co/dev/reference/op_softplus.md)
  : Softplus activation function.

- [`op_softsign()`](https://keras3.posit.co/dev/reference/op_softsign.md)
  : Softsign activation function.

- [`op_sparse_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/op_sparse_categorical_crossentropy.md)
  : Computes sparse categorical cross-entropy loss.

- [`op_sparse_plus()`](https://keras3.posit.co/dev/reference/op_sparse_plus.md)
  : SparsePlus activation function.

- [`op_sparse_sigmoid()`](https://keras3.posit.co/dev/reference/op_sparse_sigmoid.md)
  : Sparse sigmoid activation function.

- [`op_sparsemax()`](https://keras3.posit.co/dev/reference/op_sparsemax.md)
  : Sparsemax activation function.

- [`op_squareplus()`](https://keras3.posit.co/dev/reference/op_squareplus.md)
  : Squareplus activation function.

- [`op_tanh_shrink()`](https://keras3.posit.co/dev/reference/op_tanh_shrink.md)
  : Applies the tanh shrink function element-wise.

- [`op_threshold()`](https://keras3.posit.co/dev/reference/op_threshold.md)
  : Threshold activation function.

- [`op_unravel_index()`](https://keras3.posit.co/dev/reference/op_unravel_index.md)
  : Convert flat indices to coordinate arrays in a given array shape.

### Linear Algebra Operations

- [`op_cholesky()`](https://keras3.posit.co/dev/reference/op_cholesky.md)
  : Computes the Cholesky decomposition of a positive semi-definite
  matrix.

- [`op_det()`](https://keras3.posit.co/dev/reference/op_det.md) :
  Computes the determinant of a square tensor.

- [`op_eig()`](https://keras3.posit.co/dev/reference/op_eig.md) :
  Computes the eigenvalues and eigenvectors of a square matrix.

- [`op_eigh()`](https://keras3.posit.co/dev/reference/op_eigh.md) :
  Computes the eigenvalues and eigenvectors of a complex Hermitian.

- [`op_inv()`](https://keras3.posit.co/dev/reference/op_inv.md) :
  Computes the inverse of a square tensor.

- [`op_lstsq()`](https://keras3.posit.co/dev/reference/op_lstsq.md) :
  Return the least-squares solution to a linear matrix equation.

- [`op_lu_factor()`](https://keras3.posit.co/dev/reference/op_lu_factor.md)
  : Computes the lower-upper decomposition of a square matrix.

- [`op_norm()`](https://keras3.posit.co/dev/reference/op_norm.md) :
  Matrix or vector norm.

- [`op_slogdet()`](https://keras3.posit.co/dev/reference/op_slogdet.md)
  : Compute the sign and natural logarithm of the determinant of a
  matrix.

- [`op_solve_triangular()`](https://keras3.posit.co/dev/reference/op_solve_triangular.md)
  :

  Solves a linear system of equations given by `a %*% x = b`.

- [`op_svd()`](https://keras3.posit.co/dev/reference/op_svd.md) :
  Computes the singular value decomposition of a matrix.

### Image Operations

- [`op_image_affine_transform()`](https://keras3.posit.co/dev/reference/op_image_affine_transform.md)
  : Applies the given transform(s) to the image(s).

- [`op_image_crop()`](https://keras3.posit.co/dev/reference/op_image_crop.md)
  :

  Crop `images` to a specified `height` and `width`.

- [`op_image_extract_patches()`](https://keras3.posit.co/dev/reference/op_image_extract_patches.md)
  : Extracts patches from the image(s).

- [`op_image_gaussian_blur()`](https://keras3.posit.co/dev/reference/op_image_gaussian_blur.md)
  : Applies a Gaussian blur to the image(s).

- [`op_image_hsv_to_rgb()`](https://keras3.posit.co/dev/reference/op_image_hsv_to_rgb.md)
  : Convert HSV images to RGB.

- [`op_image_map_coordinates()`](https://keras3.posit.co/dev/reference/op_image_map_coordinates.md)
  : Map the input array to new coordinates by interpolation.

- [`op_image_pad()`](https://keras3.posit.co/dev/reference/op_image_pad.md)
  :

  Pad `images` with zeros to the specified `height` and `width`.

- [`op_image_perspective_transform()`](https://keras3.posit.co/dev/reference/op_image_perspective_transform.md)
  : Applies a perspective transformation to the image(s).

- [`op_image_resize()`](https://keras3.posit.co/dev/reference/op_image_resize.md)
  : Resize images to size using the specified interpolation method.

- [`op_image_rgb_to_grayscale()`](https://keras3.posit.co/dev/reference/op_image_rgb_to_grayscale.md)
  : Convert RGB images to grayscale.

- [`op_image_rgb_to_hsv()`](https://keras3.posit.co/dev/reference/op_image_rgb_to_hsv.md)
  : Convert RGB images to HSV.

## Losses

- [`loss_binary_crossentropy()`](https://keras3.posit.co/dev/reference/loss_binary_crossentropy.md)
  : Computes the cross-entropy loss between true labels and predicted
  labels.

- [`loss_binary_focal_crossentropy()`](https://keras3.posit.co/dev/reference/loss_binary_focal_crossentropy.md)
  : Computes focal cross-entropy loss between true labels and
  predictions.

- [`loss_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/loss_categorical_crossentropy.md)
  : Computes the crossentropy loss between the labels and predictions.

- [`loss_categorical_focal_crossentropy()`](https://keras3.posit.co/dev/reference/loss_categorical_focal_crossentropy.md)
  : Computes the alpha balanced focal crossentropy loss.

- [`loss_categorical_generalized_cross_entropy()`](https://keras3.posit.co/dev/reference/loss_categorical_generalized_cross_entropy.md)
  : Computes the generalized cross entropy loss.

- [`loss_categorical_hinge()`](https://keras3.posit.co/dev/reference/loss_categorical_hinge.md)
  :

  Computes the categorical hinge loss between `y_true` & `y_pred`.

- [`loss_circle()`](https://keras3.posit.co/dev/reference/loss_circle.md)
  : Computes Circle Loss between integer labels and L2-normalized
  embeddings.

- [`loss_cosine_similarity()`](https://keras3.posit.co/dev/reference/loss_cosine_similarity.md)
  :

  Computes the cosine similarity between `y_true` & `y_pred`.

- [`loss_ctc()`](https://keras3.posit.co/dev/reference/loss_ctc.md) :
  CTC (Connectionist Temporal Classification) loss.

- [`loss_dice()`](https://keras3.posit.co/dev/reference/loss_dice.md) :

  Computes the Dice loss value between `y_true` and `y_pred`.

- [`loss_hinge()`](https://keras3.posit.co/dev/reference/loss_hinge.md)
  :

  Computes the hinge loss between `y_true` & `y_pred`.

- [`loss_huber()`](https://keras3.posit.co/dev/reference/loss_huber.md)
  :

  Computes the Huber loss between `y_true` & `y_pred`.

- [`loss_kl_divergence()`](https://keras3.posit.co/dev/reference/loss_kl_divergence.md)
  :

  Computes Kullback-Leibler divergence loss between `y_true` & `y_pred`.

- [`loss_log_cosh()`](https://keras3.posit.co/dev/reference/loss_log_cosh.md)
  : Computes the logarithm of the hyperbolic cosine of the prediction
  error.

- [`loss_mean_absolute_error()`](https://keras3.posit.co/dev/reference/loss_mean_absolute_error.md)
  : Computes the mean of absolute difference between labels and
  predictions.

- [`loss_mean_absolute_percentage_error()`](https://keras3.posit.co/dev/reference/loss_mean_absolute_percentage_error.md)
  :

  Computes the mean absolute percentage error between `y_true` and
  `y_pred`.

- [`loss_mean_squared_error()`](https://keras3.posit.co/dev/reference/loss_mean_squared_error.md)
  : Computes the mean of squares of errors between labels and
  predictions.

- [`loss_mean_squared_logarithmic_error()`](https://keras3.posit.co/dev/reference/loss_mean_squared_logarithmic_error.md)
  :

  Computes the mean squared logarithmic error between `y_true` and
  `y_pred`.

- [`loss_poisson()`](https://keras3.posit.co/dev/reference/loss_poisson.md)
  :

  Computes the Poisson loss between `y_true` & `y_pred`.

- [`loss_sparse_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/loss_sparse_categorical_crossentropy.md)
  : Computes the crossentropy loss between the labels and predictions.

- [`loss_squared_hinge()`](https://keras3.posit.co/dev/reference/loss_squared_hinge.md)
  :

  Computes the squared hinge loss between `y_true` & `y_pred`.

- [`loss_tversky()`](https://keras3.posit.co/dev/reference/loss_tversky.md)
  :

  Computes the Tversky loss value between `y_true` and `y_pred`.

- [`Loss()`](https://keras3.posit.co/dev/reference/Loss.md) :

  Subclass the base `Loss` class

## Metrics

- [`metric_auc()`](https://keras3.posit.co/dev/reference/metric_auc.md)
  : Approximates the AUC (Area under the curve) of the ROC or PR curves.

- [`metric_binary_accuracy()`](https://keras3.posit.co/dev/reference/metric_binary_accuracy.md)
  : Calculates how often predictions match binary labels.

- [`metric_binary_crossentropy()`](https://keras3.posit.co/dev/reference/metric_binary_crossentropy.md)
  : Computes the crossentropy metric between the labels and predictions.

- [`metric_binary_focal_crossentropy()`](https://keras3.posit.co/dev/reference/metric_binary_focal_crossentropy.md)
  : Computes the binary focal crossentropy loss.

- [`metric_binary_iou()`](https://keras3.posit.co/dev/reference/metric_binary_iou.md)
  : Computes the Intersection-Over-Union metric for class 0 and/or 1.

- [`metric_categorical_accuracy()`](https://keras3.posit.co/dev/reference/metric_categorical_accuracy.md)
  : Calculates how often predictions match one-hot labels.

- [`metric_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/metric_categorical_crossentropy.md)
  : Computes the crossentropy metric between the labels and predictions.

- [`metric_categorical_focal_crossentropy()`](https://keras3.posit.co/dev/reference/metric_categorical_focal_crossentropy.md)
  : Computes the categorical focal crossentropy loss.

- [`metric_categorical_hinge()`](https://keras3.posit.co/dev/reference/metric_categorical_hinge.md)
  :

  Computes the categorical hinge metric between `y_true` and `y_pred`.

- [`metric_concordance_correlation()`](https://keras3.posit.co/dev/reference/metric_concordance_correlation.md)
  : Calculates the Concordance Correlation Coefficient (CCC).

- [`metric_cosine_similarity()`](https://keras3.posit.co/dev/reference/metric_cosine_similarity.md)
  : Computes the cosine similarity between the labels and predictions.

- [`metric_f1_score()`](https://keras3.posit.co/dev/reference/metric_f1_score.md)
  : Computes F-1 Score.

- [`metric_false_negatives()`](https://keras3.posit.co/dev/reference/metric_false_negatives.md)
  : Calculates the number of false negatives.

- [`metric_false_positives()`](https://keras3.posit.co/dev/reference/metric_false_positives.md)
  : Calculates the number of false positives.

- [`metric_fbeta_score()`](https://keras3.posit.co/dev/reference/metric_fbeta_score.md)
  : Computes F-Beta score.

- [`metric_hinge()`](https://keras3.posit.co/dev/reference/metric_hinge.md)
  :

  Computes the hinge metric between `y_true` and `y_pred`.

- [`metric_huber()`](https://keras3.posit.co/dev/reference/metric_huber.md)
  : Computes Huber loss value.

- [`metric_iou()`](https://keras3.posit.co/dev/reference/metric_iou.md)
  : Computes the Intersection-Over-Union metric for specific target
  classes.

- [`metric_kl_divergence()`](https://keras3.posit.co/dev/reference/metric_kl_divergence.md)
  :

  Computes Kullback-Leibler divergence metric between `y_true` and

- [`metric_log_cosh()`](https://keras3.posit.co/dev/reference/metric_log_cosh.md)
  : Logarithm of the hyperbolic cosine of the prediction error.

- [`metric_log_cosh_error()`](https://keras3.posit.co/dev/reference/metric_log_cosh_error.md)
  : Computes the logarithm of the hyperbolic cosine of the prediction
  error.

- [`metric_mean()`](https://keras3.posit.co/dev/reference/metric_mean.md)
  : Compute the (weighted) mean of the given values.

- [`metric_mean_absolute_error()`](https://keras3.posit.co/dev/reference/metric_mean_absolute_error.md)
  : Computes the mean absolute error between the labels and predictions.

- [`metric_mean_absolute_percentage_error()`](https://keras3.posit.co/dev/reference/metric_mean_absolute_percentage_error.md)
  :

  Computes mean absolute percentage error between `y_true` and `y_pred`.

- [`metric_mean_iou()`](https://keras3.posit.co/dev/reference/metric_mean_iou.md)
  : Computes the mean Intersection-Over-Union metric.

- [`metric_mean_squared_error()`](https://keras3.posit.co/dev/reference/metric_mean_squared_error.md)
  :

  Computes the mean squared error between `y_true` and `y_pred`.

- [`metric_mean_squared_logarithmic_error()`](https://keras3.posit.co/dev/reference/metric_mean_squared_logarithmic_error.md)
  :

  Computes mean squared logarithmic error between `y_true` and `y_pred`.

- [`metric_mean_wrapper()`](https://keras3.posit.co/dev/reference/metric_mean_wrapper.md)
  :

  Wrap a stateless metric function with the `Mean` metric.

- [`metric_one_hot_iou()`](https://keras3.posit.co/dev/reference/metric_one_hot_iou.md)
  : Computes the Intersection-Over-Union metric for one-hot encoded
  labels.

- [`metric_one_hot_mean_iou()`](https://keras3.posit.co/dev/reference/metric_one_hot_mean_iou.md)
  : Computes mean Intersection-Over-Union metric for one-hot encoded
  labels.

- [`metric_pearson_correlation()`](https://keras3.posit.co/dev/reference/metric_pearson_correlation.md)
  : Calculates the Pearson Correlation Coefficient (PCC).

- [`metric_poisson()`](https://keras3.posit.co/dev/reference/metric_poisson.md)
  :

  Computes the Poisson metric between `y_true` and `y_pred`.

- [`metric_precision()`](https://keras3.posit.co/dev/reference/metric_precision.md)
  : Computes the precision of the predictions with respect to the
  labels.

- [`metric_precision_at_recall()`](https://keras3.posit.co/dev/reference/metric_precision_at_recall.md)
  : Computes best precision where recall is \>= specified value.

- [`metric_r2_score()`](https://keras3.posit.co/dev/reference/metric_r2_score.md)
  : Computes R2 score.

- [`metric_recall()`](https://keras3.posit.co/dev/reference/metric_recall.md)
  : Computes the recall of the predictions with respect to the labels.

- [`metric_recall_at_precision()`](https://keras3.posit.co/dev/reference/metric_recall_at_precision.md)
  : Computes best recall where precision is \>= specified value.

- [`metric_root_mean_squared_error()`](https://keras3.posit.co/dev/reference/metric_root_mean_squared_error.md)
  :

  Computes root mean squared error metric between `y_true` and `y_pred`.

- [`metric_sensitivity_at_specificity()`](https://keras3.posit.co/dev/reference/metric_sensitivity_at_specificity.md)
  : Computes best sensitivity where specificity is \>= specified value.

- [`metric_sparse_categorical_accuracy()`](https://keras3.posit.co/dev/reference/metric_sparse_categorical_accuracy.md)
  : Calculates how often predictions match integer labels.

- [`metric_sparse_categorical_crossentropy()`](https://keras3.posit.co/dev/reference/metric_sparse_categorical_crossentropy.md)
  : Computes the crossentropy metric between the labels and predictions.

- [`metric_sparse_top_k_categorical_accuracy()`](https://keras3.posit.co/dev/reference/metric_sparse_top_k_categorical_accuracy.md)
  :

  Computes how often integer targets are in the top `K` predictions.

- [`metric_specificity_at_sensitivity()`](https://keras3.posit.co/dev/reference/metric_specificity_at_sensitivity.md)
  : Computes best specificity where sensitivity is \>= specified value.

- [`metric_squared_hinge()`](https://keras3.posit.co/dev/reference/metric_squared_hinge.md)
  :

  Computes the hinge metric between `y_true` and `y_pred`.

- [`metric_sum()`](https://keras3.posit.co/dev/reference/metric_sum.md)
  : Compute the (weighted) sum of the given values.

- [`metric_top_k_categorical_accuracy()`](https://keras3.posit.co/dev/reference/metric_top_k_categorical_accuracy.md)
  :

  Computes how often targets are in the top `K` predictions.

- [`metric_true_negatives()`](https://keras3.posit.co/dev/reference/metric_true_negatives.md)
  : Calculates the number of true negatives.

- [`metric_true_positives()`](https://keras3.posit.co/dev/reference/metric_true_positives.md)
  : Calculates the number of true positives.

- [`custom_metric()`](https://keras3.posit.co/dev/reference/custom_metric.md)
  : Custom metric function

- [`reset_state()`](https://keras3.posit.co/dev/reference/reset_state.md)
  : Reset the state for a model, layer or metric.

- [`Metric()`](https://keras3.posit.co/dev/reference/Metric.md) :

  Subclass the base `Metric` class

## Data Loading

Keras data loading utilities help you quickly go from raw data to a TF
`Dataset` object that can be used to efficiently train a model. These
loading utilites can be combined with preprocessing layers to futher
transform your input dataset before training.

- [`image_dataset_from_directory()`](https://keras3.posit.co/dev/reference/image_dataset_from_directory.md)
  :

  Generates a `tf.data.Dataset` from image files in a directory.

- [`text_dataset_from_directory()`](https://keras3.posit.co/dev/reference/text_dataset_from_directory.md)
  :

  Generates a `tf.data.Dataset` from text files in a directory.

- [`audio_dataset_from_directory()`](https://keras3.posit.co/dev/reference/audio_dataset_from_directory.md)
  :

  Generates a `tf.data.Dataset` from audio files in a directory.

- [`timeseries_dataset_from_array()`](https://keras3.posit.co/dev/reference/timeseries_dataset_from_array.md)
  : Creates a dataset of sliding windows over a timeseries provided as
  array.

## Preprocessing

- [`layer_feature_space()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_cross()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_custom()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_float()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_float_rescaled()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_float_normalized()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_float_discretized()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_integer_categorical()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_string_categorical()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_string_hashed()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  [`feature_integer_hashed()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)
  : One-stop utility for preprocessing and encoding structured data.
- [`adapt()`](https://keras3.posit.co/dev/reference/adapt.md) : Fits the
  state of the preprocessing layer to the data being passed

### Numerical Features Preprocessing Layers

- [`layer_normalization()`](https://keras3.posit.co/dev/reference/layer_normalization.md)
  : A preprocessing layer that normalizes continuous features.
- [`layer_discretization()`](https://keras3.posit.co/dev/reference/layer_discretization.md)
  : A preprocessing layer which buckets continuous features by ranges.

### Categorical Features Preprocessing Layers

- [`layer_category_encoding()`](https://keras3.posit.co/dev/reference/layer_category_encoding.md)
  : A preprocessing layer which encodes integer features.
- [`layer_hashing()`](https://keras3.posit.co/dev/reference/layer_hashing.md)
  : A preprocessing layer which hashes and bins categorical features.
- [`layer_hashed_crossing()`](https://keras3.posit.co/dev/reference/layer_hashed_crossing.md)
  : A preprocessing layer which crosses features using the "hashing
  trick".
- [`layer_string_lookup()`](https://keras3.posit.co/dev/reference/layer_string_lookup.md)
  : A preprocessing layer that maps strings to (possibly encoded)
  indices.
- [`layer_integer_lookup()`](https://keras3.posit.co/dev/reference/layer_integer_lookup.md)
  : A preprocessing layer that maps integers to (possibly encoded)
  indices.

### Text Preprocessing Layers

- [`layer_text_vectorization()`](https://keras3.posit.co/dev/reference/layer_text_vectorization.md)
  [`get_vocabulary()`](https://keras3.posit.co/dev/reference/layer_text_vectorization.md)
  [`set_vocabulary()`](https://keras3.posit.co/dev/reference/layer_text_vectorization.md)
  : A preprocessing layer which maps text features to integer sequences.

### Sequence Preprocessing

- [`timeseries_dataset_from_array()`](https://keras3.posit.co/dev/reference/timeseries_dataset_from_array.md)
  : Creates a dataset of sliding windows over a timeseries provided as
  array.
- [`pad_sequences()`](https://keras3.posit.co/dev/reference/pad_sequences.md)
  : Pads sequences to the same length.

### Image Preprocessing Layers

- [`layer_resizing()`](https://keras3.posit.co/dev/reference/layer_resizing.md)
  : A preprocessing layer which resizes images.
- [`layer_rescaling()`](https://keras3.posit.co/dev/reference/layer_rescaling.md)
  : A preprocessing layer which rescales input values to a new range.
- [`layer_center_crop()`](https://keras3.posit.co/dev/reference/layer_center_crop.md)
  : A preprocessing layer which crops images.

## Image Preprocessing

- [`image_array_save()`](https://keras3.posit.co/dev/reference/image_array_save.md)
  : Saves an image stored as an array to a path or file object.

- [`image_dataset_from_directory()`](https://keras3.posit.co/dev/reference/image_dataset_from_directory.md)
  :

  Generates a `tf.data.Dataset` from image files in a directory.

- [`image_from_array()`](https://keras3.posit.co/dev/reference/image_from_array.md)
  : Converts a 3D array to a PIL Image instance.

- [`image_load()`](https://keras3.posit.co/dev/reference/image_load.md)
  : Loads an image into PIL format.

- [`image_smart_resize()`](https://keras3.posit.co/dev/reference/image_smart_resize.md)
  : Resize images to a target size without aspect ratio distortion.

- [`image_to_array()`](https://keras3.posit.co/dev/reference/image_to_array.md)
  : Converts a PIL Image instance to a matrix.

- [`op_image_affine_transform()`](https://keras3.posit.co/dev/reference/op_image_affine_transform.md)
  : Applies the given transform(s) to the image(s).

- [`op_image_crop()`](https://keras3.posit.co/dev/reference/op_image_crop.md)
  :

  Crop `images` to a specified `height` and `width`.

- [`op_image_elastic_transform()`](https://keras3.posit.co/dev/reference/op_image_elastic_transform.md)
  : Applies elastic deformation to the image(s).

- [`op_image_extract_patches()`](https://keras3.posit.co/dev/reference/op_image_extract_patches.md)
  : Extracts patches from the image(s).

- [`op_image_gaussian_blur()`](https://keras3.posit.co/dev/reference/op_image_gaussian_blur.md)
  : Applies a Gaussian blur to the image(s).

- [`op_image_hsv_to_rgb()`](https://keras3.posit.co/dev/reference/op_image_hsv_to_rgb.md)
  : Convert HSV images to RGB.

- [`op_image_map_coordinates()`](https://keras3.posit.co/dev/reference/op_image_map_coordinates.md)
  : Map the input array to new coordinates by interpolation.

- [`op_image_pad()`](https://keras3.posit.co/dev/reference/op_image_pad.md)
  :

  Pad `images` with zeros to the specified `height` and `width`.

- [`op_image_perspective_transform()`](https://keras3.posit.co/dev/reference/op_image_perspective_transform.md)
  : Applies a perspective transformation to the image(s).

- [`op_image_resize()`](https://keras3.posit.co/dev/reference/op_image_resize.md)
  : Resize images to size using the specified interpolation method.

- [`op_image_rgb_to_grayscale()`](https://keras3.posit.co/dev/reference/op_image_rgb_to_grayscale.md)
  : Convert RGB images to grayscale.

- [`op_image_rgb_to_hsv()`](https://keras3.posit.co/dev/reference/op_image_rgb_to_hsv.md)
  : Convert RGB images to HSV.

### Image augmentation Layers

- [`layer_random_crop()`](https://keras3.posit.co/dev/reference/layer_random_crop.md)
  : A preprocessing layer which randomly crops images during training.
- [`layer_random_flip()`](https://keras3.posit.co/dev/reference/layer_random_flip.md)
  : A preprocessing layer which randomly flips images during training.
- [`layer_random_translation()`](https://keras3.posit.co/dev/reference/layer_random_translation.md)
  : A preprocessing layer which randomly translates images during
  training.
- [`layer_random_rotation()`](https://keras3.posit.co/dev/reference/layer_random_rotation.md)
  : A preprocessing layer which randomly rotates images during training.
- [`layer_random_zoom()`](https://keras3.posit.co/dev/reference/layer_random_zoom.md)
  : A preprocessing layer which randomly zooms images during training.
- [`layer_random_contrast()`](https://keras3.posit.co/dev/reference/layer_random_contrast.md)
  : A preprocessing layer which randomly adjusts contrast during
  training.
- [`layer_random_brightness()`](https://keras3.posit.co/dev/reference/layer_random_brightness.md)
  : A preprocessing layer which randomly adjusts brightness during
  training.

### Application Preprocessing

- [`application_preprocess_inputs()`](https://keras3.posit.co/dev/reference/process_utils.md)
  [`application_decode_predictions()`](https://keras3.posit.co/dev/reference/process_utils.md)
  : Preprocessing and postprocessing utilities

## Optimizers

- [`optimizer_adadelta()`](https://keras3.posit.co/dev/reference/optimizer_adadelta.md)
  : Optimizer that implements the Adadelta algorithm.
- [`optimizer_adafactor()`](https://keras3.posit.co/dev/reference/optimizer_adafactor.md)
  : Optimizer that implements the Adafactor algorithm.
- [`optimizer_adagrad()`](https://keras3.posit.co/dev/reference/optimizer_adagrad.md)
  : Optimizer that implements the Adagrad algorithm.
- [`optimizer_adam()`](https://keras3.posit.co/dev/reference/optimizer_adam.md)
  : Optimizer that implements the Adam algorithm.
- [`optimizer_adam_w()`](https://keras3.posit.co/dev/reference/optimizer_adam_w.md)
  : Optimizer that implements the AdamW algorithm.
- [`optimizer_adamax()`](https://keras3.posit.co/dev/reference/optimizer_adamax.md)
  : Optimizer that implements the Adamax algorithm.
- [`optimizer_ftrl()`](https://keras3.posit.co/dev/reference/optimizer_ftrl.md)
  : Optimizer that implements the FTRL algorithm.
- [`optimizer_lamb()`](https://keras3.posit.co/dev/reference/optimizer_lamb.md)
  : Optimizer that implements the Lamb algorithm.
- [`optimizer_lion()`](https://keras3.posit.co/dev/reference/optimizer_lion.md)
  : Optimizer that implements the Lion algorithm.
- [`optimizer_loss_scale()`](https://keras3.posit.co/dev/reference/optimizer_loss_scale.md)
  : An optimizer that dynamically scales the loss to prevent underflow.
- [`optimizer_muon()`](https://keras3.posit.co/dev/reference/optimizer_muon.md)
  : Optimizer that implements the Muon algorithm.
- [`optimizer_nadam()`](https://keras3.posit.co/dev/reference/optimizer_nadam.md)
  : Optimizer that implements the Nadam algorithm.
- [`optimizer_rmsprop()`](https://keras3.posit.co/dev/reference/optimizer_rmsprop.md)
  : Optimizer that implements the RMSprop algorithm.
- [`optimizer_sgd()`](https://keras3.posit.co/dev/reference/optimizer_sgd.md)
  : Gradient descent (with momentum) optimizer.

## Learning Rate Schedules

- [`learning_rate_schedule_cosine_decay()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_cosine_decay.md)
  :

  A `LearningRateSchedule` that uses a cosine decay with optional
  warmup.

- [`learning_rate_schedule_cosine_decay_restarts()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_cosine_decay_restarts.md)
  :

  A `LearningRateSchedule` that uses a cosine decay schedule with
  restarts.

- [`learning_rate_schedule_exponential_decay()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_exponential_decay.md)
  :

  A `LearningRateSchedule` that uses an exponential decay schedule.

- [`learning_rate_schedule_inverse_time_decay()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_inverse_time_decay.md)
  :

  A `LearningRateSchedule` that uses an inverse time decay schedule.

- [`learning_rate_schedule_piecewise_constant_decay()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_piecewise_constant_decay.md)
  :

  A `LearningRateSchedule` that uses a piecewise constant decay
  schedule.

- [`learning_rate_schedule_polynomial_decay()`](https://keras3.posit.co/dev/reference/learning_rate_schedule_polynomial_decay.md)
  :

  A `LearningRateSchedule` that uses a polynomial decay schedule.

- [`LearningRateSchedule()`](https://keras3.posit.co/dev/reference/LearningRateSchedule.md)
  :

  Define a custom `LearningRateSchedule` class

## Initializers

- [`initializer_constant()`](https://keras3.posit.co/dev/reference/initializer_constant.md)
  : Initializer that generates tensors with constant values.
- [`initializer_glorot_normal()`](https://keras3.posit.co/dev/reference/initializer_glorot_normal.md)
  : The Glorot normal initializer, also called Xavier normal
  initializer.
- [`initializer_glorot_uniform()`](https://keras3.posit.co/dev/reference/initializer_glorot_uniform.md)
  : The Glorot uniform initializer, also called Xavier uniform
  initializer.
- [`initializer_he_normal()`](https://keras3.posit.co/dev/reference/initializer_he_normal.md)
  : He normal initializer.
- [`initializer_he_uniform()`](https://keras3.posit.co/dev/reference/initializer_he_uniform.md)
  : He uniform variance scaling initializer.
- [`initializer_identity()`](https://keras3.posit.co/dev/reference/initializer_identity.md)
  : Initializer that generates the identity matrix.
- [`initializer_lecun_normal()`](https://keras3.posit.co/dev/reference/initializer_lecun_normal.md)
  : Lecun normal initializer.
- [`initializer_lecun_uniform()`](https://keras3.posit.co/dev/reference/initializer_lecun_uniform.md)
  : Lecun uniform initializer.
- [`initializer_ones()`](https://keras3.posit.co/dev/reference/initializer_ones.md)
  : Initializer that generates tensors initialized to 1.
- [`initializer_orthogonal()`](https://keras3.posit.co/dev/reference/initializer_orthogonal.md)
  : Initializer that generates an orthogonal matrix.
- [`initializer_random_normal()`](https://keras3.posit.co/dev/reference/initializer_random_normal.md)
  : Random normal initializer.
- [`initializer_random_uniform()`](https://keras3.posit.co/dev/reference/initializer_random_uniform.md)
  : Random uniform initializer.
- [`initializer_stft()`](https://keras3.posit.co/dev/reference/initializer_stft.md)
  : Initializer of Conv kernels for Short-term Fourier Transformation
  (STFT).
- [`initializer_truncated_normal()`](https://keras3.posit.co/dev/reference/initializer_truncated_normal.md)
  : Initializer that generates a truncated normal distribution.
- [`initializer_variance_scaling()`](https://keras3.posit.co/dev/reference/initializer_variance_scaling.md)
  : Initializer that adapts its scale to the shape of its input tensors.
- [`initializer_zeros()`](https://keras3.posit.co/dev/reference/initializer_zeros.md)
  : Initializer that generates tensors initialized to 0.

## Constraints

- [`Constraint()`](https://keras3.posit.co/dev/reference/Constraint.md)
  :

  Define a custom `Constraint` class

- [`constraint_maxnorm()`](https://keras3.posit.co/dev/reference/constraint_maxnorm.md)
  : MaxNorm weight constraint.

- [`constraint_minmaxnorm()`](https://keras3.posit.co/dev/reference/constraint_minmaxnorm.md)
  : MinMaxNorm weight constraint.

- [`constraint_nonneg()`](https://keras3.posit.co/dev/reference/constraint_nonneg.md)
  : Constrains the weights to be non-negative.

- [`constraint_unitnorm()`](https://keras3.posit.co/dev/reference/constraint_unitnorm.md)
  : Constrains the weights incident to each hidden unit to have unit
  norm.

## Regularizers

- [`regularizer_l1()`](https://keras3.posit.co/dev/reference/regularizer_l1.md)
  : A regularizer that applies a L1 regularization penalty.
- [`regularizer_l1_l2()`](https://keras3.posit.co/dev/reference/regularizer_l1_l2.md)
  : A regularizer that applies both L1 and L2 regularization penalties.
- [`regularizer_l2()`](https://keras3.posit.co/dev/reference/regularizer_l2.md)
  : A regularizer that applies a L2 regularization penalty.
- [`regularizer_orthogonal()`](https://keras3.posit.co/dev/reference/regularizer_orthogonal.md)
  : Regularizer that encourages input vectors to be orthogonal to each
  other.

## Activations

- [`activation_celu()`](https://keras3.posit.co/dev/reference/activation_celu.md)
  : Continuously Differentiable Exponential Linear Unit.
- [`activation_elu()`](https://keras3.posit.co/dev/reference/activation_elu.md)
  : Exponential Linear Unit.
- [`activation_exponential()`](https://keras3.posit.co/dev/reference/activation_exponential.md)
  : Exponential activation function.
- [`activation_gelu()`](https://keras3.posit.co/dev/reference/activation_gelu.md)
  : Gaussian error linear unit (GELU) activation function.
- [`activation_glu()`](https://keras3.posit.co/dev/reference/activation_glu.md)
  : Gated Linear Unit (GLU) activation function.
- [`activation_hard_shrink()`](https://keras3.posit.co/dev/reference/activation_hard_shrink.md)
  : Hard Shrink activation function.
- [`activation_hard_sigmoid()`](https://keras3.posit.co/dev/reference/activation_hard_sigmoid.md)
  : Hard sigmoid activation function.
- [`activation_hard_silu()`](https://keras3.posit.co/dev/reference/activation_hard_silu.md)
  [`activation_hard_swish()`](https://keras3.posit.co/dev/reference/activation_hard_silu.md)
  : Hard SiLU activation function, also known as Hard Swish.
- [`activation_hard_tanh()`](https://keras3.posit.co/dev/reference/activation_hard_tanh.md)
  : HardTanh activation function.
- [`activation_leaky_relu()`](https://keras3.posit.co/dev/reference/activation_leaky_relu.md)
  : Leaky relu activation function.
- [`activation_linear()`](https://keras3.posit.co/dev/reference/activation_linear.md)
  : Linear activation function (pass-through).
- [`activation_log_sigmoid()`](https://keras3.posit.co/dev/reference/activation_log_sigmoid.md)
  : Logarithm of the sigmoid activation function.
- [`activation_log_softmax()`](https://keras3.posit.co/dev/reference/activation_log_softmax.md)
  : Log-Softmax activation function.
- [`activation_mish()`](https://keras3.posit.co/dev/reference/activation_mish.md)
  : Mish activation function.
- [`activation_relu()`](https://keras3.posit.co/dev/reference/activation_relu.md)
  : Applies the rectified linear unit activation function.
- [`activation_relu6()`](https://keras3.posit.co/dev/reference/activation_relu6.md)
  : Relu6 activation function.
- [`activation_selu()`](https://keras3.posit.co/dev/reference/activation_selu.md)
  : Scaled Exponential Linear Unit (SELU).
- [`activation_sigmoid()`](https://keras3.posit.co/dev/reference/activation_sigmoid.md)
  : Sigmoid activation function.
- [`activation_silu()`](https://keras3.posit.co/dev/reference/activation_silu.md)
  : Swish (or Silu) activation function.
- [`activation_soft_shrink()`](https://keras3.posit.co/dev/reference/activation_soft_shrink.md)
  : Soft Shrink activation function.
- [`activation_softmax()`](https://keras3.posit.co/dev/reference/activation_softmax.md)
  : Softmax converts a vector of values to a probability distribution.
- [`activation_softplus()`](https://keras3.posit.co/dev/reference/activation_softplus.md)
  : Softplus activation function.
- [`activation_softsign()`](https://keras3.posit.co/dev/reference/activation_softsign.md)
  : Softsign activation function.
- [`activation_sparse_plus()`](https://keras3.posit.co/dev/reference/activation_sparse_plus.md)
  : SparsePlus activation function.
- [`activation_sparse_sigmoid()`](https://keras3.posit.co/dev/reference/activation_sparse_sigmoid.md)
  : Sparse sigmoid activation function.
- [`activation_sparsemax()`](https://keras3.posit.co/dev/reference/activation_sparsemax.md)
  : Sparsemax activation function.
- [`activation_squareplus()`](https://keras3.posit.co/dev/reference/activation_squareplus.md)
  : Squareplus activation function.
- [`activation_tanh()`](https://keras3.posit.co/dev/reference/activation_tanh.md)
  : Hyperbolic tangent activation function.
- [`activation_tanh_shrink()`](https://keras3.posit.co/dev/reference/activation_tanh_shrink.md)
  : Tanh shrink activation function.
- [`activation_threshold()`](https://keras3.posit.co/dev/reference/activation_threshold.md)
  : Threshold activation function.

## Random Tensor Generators

- [`random_uniform()`](https://keras3.posit.co/dev/reference/random_uniform.md)
  : Draw samples from a uniform distribution.
- [`random_normal()`](https://keras3.posit.co/dev/reference/random_normal.md)
  : Draw random samples from a normal (Gaussian) distribution.
- [`random_truncated_normal()`](https://keras3.posit.co/dev/reference/random_truncated_normal.md)
  : Draw samples from a truncated normal distribution.
- [`random_gamma()`](https://keras3.posit.co/dev/reference/random_gamma.md)
  : Draw random samples from the Gamma distribution.
- [`random_categorical()`](https://keras3.posit.co/dev/reference/random_categorical.md)
  : Draws samples from a categorical distribution.
- [`random_integer()`](https://keras3.posit.co/dev/reference/random_integer.md)
  : Draw random integers from a uniform distribution.
- [`random_dropout()`](https://keras3.posit.co/dev/reference/random_dropout.md)
  : Randomly set some values in a tensor to 0.
- [`random_shuffle()`](https://keras3.posit.co/dev/reference/random_shuffle.md)
  : Shuffle the elements of a tensor uniformly at random along an axis.
- [`random_beta()`](https://keras3.posit.co/dev/reference/random_beta.md)
  : Draw samples from a Beta distribution.
- [`random_binomial()`](https://keras3.posit.co/dev/reference/random_binomial.md)
  : Draw samples from a Binomial distribution.
- [`random_seed_generator()`](https://keras3.posit.co/dev/reference/random_seed_generator.md)
  : Generates variable seeds upon each call to a function generating
  random numbers.

## Builtin small datasets

- [`dataset_boston_housing()`](https://keras3.posit.co/dev/reference/dataset_boston_housing.md)
  : Boston housing price regression dataset
- [`dataset_california_housing()`](https://keras3.posit.co/dev/reference/dataset_california_housing.md)
  : Loads the California Housing dataset.
- [`dataset_cifar10()`](https://keras3.posit.co/dev/reference/dataset_cifar10.md)
  : CIFAR10 small image classification
- [`dataset_cifar100()`](https://keras3.posit.co/dev/reference/dataset_cifar100.md)
  : CIFAR100 small image classification
- [`dataset_fashion_mnist()`](https://keras3.posit.co/dev/reference/dataset_fashion_mnist.md)
  : Fashion-MNIST database of fashion articles
- [`dataset_imdb()`](https://keras3.posit.co/dev/reference/dataset_imdb.md)
  [`dataset_imdb_word_index()`](https://keras3.posit.co/dev/reference/dataset_imdb.md)
  : IMDB Movie reviews sentiment classification
- [`dataset_mnist()`](https://keras3.posit.co/dev/reference/dataset_mnist.md)
  : MNIST database of handwritten digits
- [`dataset_reuters()`](https://keras3.posit.co/dev/reference/dataset_reuters.md)
  [`dataset_reuters_word_index()`](https://keras3.posit.co/dev/reference/dataset_reuters.md)
  : Reuters newswire topics classification

## Configuration

- [`config_backend()`](https://keras3.posit.co/dev/reference/config_backend.md)
  : Publicly accessible method for determining the current backend.
- [`config_disable_flash_attention()`](https://keras3.posit.co/dev/reference/config_disable_flash_attention.md)
  : Disable flash attention.
- [`config_disable_interactive_logging()`](https://keras3.posit.co/dev/reference/config_disable_interactive_logging.md)
  : Turn off interactive logging.
- [`config_disable_traceback_filtering()`](https://keras3.posit.co/dev/reference/config_disable_traceback_filtering.md)
  : Turn off traceback filtering.
- [`config_dtype_policy()`](https://keras3.posit.co/dev/reference/config_dtype_policy.md)
  : Returns the current default dtype policy object.
- [`config_enable_flash_attention()`](https://keras3.posit.co/dev/reference/config_enable_flash_attention.md)
  : Enable flash attention.
- [`config_enable_interactive_logging()`](https://keras3.posit.co/dev/reference/config_enable_interactive_logging.md)
  : Turn on interactive logging.
- [`config_enable_traceback_filtering()`](https://keras3.posit.co/dev/reference/config_enable_traceback_filtering.md)
  : Turn on traceback filtering.
- [`config_enable_unsafe_deserialization()`](https://keras3.posit.co/dev/reference/config_enable_unsafe_deserialization.md)
  : Disables safe mode globally, allowing deserialization of lambdas.
- [`config_epsilon()`](https://keras3.posit.co/dev/reference/config_epsilon.md)
  : Return the value of the fuzz factor used in numeric expressions.
- [`config_floatx()`](https://keras3.posit.co/dev/reference/config_floatx.md)
  : Return the default float type, as a string.
- [`config_image_data_format()`](https://keras3.posit.co/dev/reference/config_image_data_format.md)
  : Return the default image data format convention.
- [`config_is_flash_attention_enabled()`](https://keras3.posit.co/dev/reference/config_is_flash_attention_enabled.md)
  : Checks whether flash attention is globally enabled in Keras.
- [`config_is_interactive_logging_enabled()`](https://keras3.posit.co/dev/reference/config_is_interactive_logging_enabled.md)
  : Check if interactive logging is enabled.
- [`config_is_nnx_enabled()`](https://keras3.posit.co/dev/reference/config_is_nnx_enabled.md)
  : Check whether NNX-specific features are enabled on the JAX backend.
- [`config_is_traceback_filtering_enabled()`](https://keras3.posit.co/dev/reference/config_is_traceback_filtering_enabled.md)
  : Check if traceback filtering is enabled.
- [`config_max_epochs()`](https://keras3.posit.co/dev/reference/config_max_epochs.md)
  [`config_set_max_epochs()`](https://keras3.posit.co/dev/reference/config_max_epochs.md)
  [`config_max_steps_per_epoch()`](https://keras3.posit.co/dev/reference/config_max_epochs.md)
  [`config_set_max_steps_per_epoch()`](https://keras3.posit.co/dev/reference/config_max_epochs.md)
  : Configure the default training loop limits.
- [`config_set_backend()`](https://keras3.posit.co/dev/reference/config_set_backend.md)
  : Reload the backend (and the Keras package).
- [`config_set_dtype_policy()`](https://keras3.posit.co/dev/reference/config_set_dtype_policy.md)
  : Sets the default dtype policy globally.
- [`config_set_epsilon()`](https://keras3.posit.co/dev/reference/config_set_epsilon.md)
  : Set the value of the fuzz factor used in numeric expressions.
- [`config_set_floatx()`](https://keras3.posit.co/dev/reference/config_set_floatx.md)
  : Set the default float dtype.
- [`config_set_image_data_format()`](https://keras3.posit.co/dev/reference/config_set_image_data_format.md)
  : Set the value of the image data format convention.

## Utils

- [`install_keras()`](https://keras3.posit.co/dev/reference/install_keras.md)
  : Install Keras

- [`use_backend()`](https://keras3.posit.co/dev/reference/use_backend.md)
  : Configure a Keras backend

- [`shape()`](https://keras3.posit.co/dev/reference/shape.md)
  [`format(`*`<keras_shape>`*`)`](https://keras3.posit.co/dev/reference/shape.md)
  [`print(`*`<keras_shape>`*`)`](https://keras3.posit.co/dev/reference/shape.md)
  [`` `[`( ``*`<keras_shape>`*`)`](https://keras3.posit.co/dev/reference/shape.md)
  [`as.integer(`*`<keras_shape>`*`)`](https://keras3.posit.co/dev/reference/shape.md)
  [`Summary(`*`<keras_shape>`*`)`](https://keras3.posit.co/dev/reference/shape.md)
  [`as.list(`*`<keras_shape>`*`)`](https://keras3.posit.co/dev/reference/shape.md)
  [`` `==`( ``*`<keras_shape>`*`)`](https://keras3.posit.co/dev/reference/shape.md)
  [`` `!=`( ``*`<keras_shape>`*`)`](https://keras3.posit.co/dev/reference/shape.md)
  : Tensor shape utility

- [`set_random_seed()`](https://keras3.posit.co/dev/reference/set_random_seed.md)
  : Sets all random seeds (Python, NumPy, and backend framework, e.g.
  TF).

- [`clear_session()`](https://keras3.posit.co/dev/reference/clear_session.md)
  : Resets all state generated by Keras.

- [`get_source_inputs()`](https://keras3.posit.co/dev/reference/get_source_inputs.md)
  :

  Returns the list of input tensors necessary to compute `tensor`.

- [`keras`](https://keras3.posit.co/dev/reference/keras.md) : Main Keras
  module

### Numerical Utils

- [`normalize()`](https://keras3.posit.co/dev/reference/normalize.md) :
  Normalizes an array.
- [`to_categorical()`](https://keras3.posit.co/dev/reference/to_categorical.md)
  : Converts a class vector (integers) to binary class matrix.

### Data Utils

- [`zip_lists()`](https://keras3.posit.co/dev/reference/zip_lists.md) :
  Zip lists
- [`get_file()`](https://keras3.posit.co/dev/reference/get_file.md) :
  Downloads a file from a URL if it not already in the cache.
- [`split_dataset()`](https://keras3.posit.co/dev/reference/split_dataset.md)
  : Splits a dataset into a left half and a right half (e.g. train /
  test).
- [`named_list()`](https://keras3.posit.co/dev/reference/named_list.md)
  : Create a named list from arguments
- [`newaxis`](https://keras3.posit.co/dev/reference/newaxis.md) : New
  axis

### Serialization Utils

- [`register_keras_serializable()`](https://keras3.posit.co/dev/reference/register_keras_serializable.md)
  : Registers a custom object with the Keras serialization framework.

- [`get_custom_objects()`](https://keras3.posit.co/dev/reference/get_custom_objects.md)
  [`set_custom_objects()`](https://keras3.posit.co/dev/reference/get_custom_objects.md)
  : Get/set the currently registered custom objects.

- [`get_registered_name()`](https://keras3.posit.co/dev/reference/get_registered_name.md)
  : Returns the name registered to an object within the Keras framework.

- [`get_registered_object()`](https://keras3.posit.co/dev/reference/get_registered_object.md)
  :

  Returns the class associated with `name` if it is registered with
  Keras.

- [`serialize_keras_object()`](https://keras3.posit.co/dev/reference/serialize_keras_object.md)
  : Retrieve the full config by serializing the Keras object.

- [`deserialize_keras_object()`](https://keras3.posit.co/dev/reference/deserialize_keras_object.md)
  : Retrieve the object by deserializing the config dict.

- [`with_custom_object_scope()`](https://keras3.posit.co/dev/reference/with_custom_object_scope.md)
  : Provide a scope with mappings of names to custom objects

- [`config_enable_unsafe_deserialization()`](https://keras3.posit.co/dev/reference/config_enable_unsafe_deserialization.md)
  : Disables safe mode globally, allowing deserialization of lambdas.

## Base Keras Classes

Define custom object by subclassing base Keras classes.

- [`Layer()`](https://keras3.posit.co/dev/reference/Layer.md) :

  Define a custom `Layer` class.

- [`Loss()`](https://keras3.posit.co/dev/reference/Loss.md) :

  Subclass the base `Loss` class

- [`Metric()`](https://keras3.posit.co/dev/reference/Metric.md) :

  Subclass the base `Metric` class

- [`Callback()`](https://keras3.posit.co/dev/reference/Callback.md) :

  Define a custom `Callback` class

- [`Constraint()`](https://keras3.posit.co/dev/reference/Constraint.md)
  :

  Define a custom `Constraint` class

- [`Model()`](https://keras3.posit.co/dev/reference/Model.md) :

  Subclass the base Keras `Model` Class

- [`LearningRateSchedule()`](https://keras3.posit.co/dev/reference/LearningRateSchedule.md)
  :

  Define a custom `LearningRateSchedule` class

- [`active_property()`](https://keras3.posit.co/dev/reference/active_property.md)
  : Create an active property class method

## Applications

### Application utilities

- [`application_preprocess_inputs()`](https://keras3.posit.co/dev/reference/process_utils.md)
  [`application_decode_predictions()`](https://keras3.posit.co/dev/reference/process_utils.md)
  : Preprocessing and postprocessing utilities

### ConvNeXt Applications

- [`application_convnext_base()`](https://keras3.posit.co/dev/reference/application_convnext_base.md)
  : Instantiates the ConvNeXtBase architecture.
- [`application_convnext_large()`](https://keras3.posit.co/dev/reference/application_convnext_large.md)
  : Instantiates the ConvNeXtLarge architecture.
- [`application_convnext_small()`](https://keras3.posit.co/dev/reference/application_convnext_small.md)
  : Instantiates the ConvNeXtSmall architecture.
- [`application_convnext_tiny()`](https://keras3.posit.co/dev/reference/application_convnext_tiny.md)
  : Instantiates the ConvNeXtTiny architecture.
- [`application_convnext_xlarge()`](https://keras3.posit.co/dev/reference/application_convnext_xlarge.md)
  : Instantiates the ConvNeXtXLarge architecture.

### Densenet Applications

- [`application_densenet121()`](https://keras3.posit.co/dev/reference/application_densenet121.md)
  : Instantiates the Densenet121 architecture.
- [`application_densenet169()`](https://keras3.posit.co/dev/reference/application_densenet169.md)
  : Instantiates the Densenet169 architecture.
- [`application_densenet201()`](https://keras3.posit.co/dev/reference/application_densenet201.md)
  : Instantiates the Densenet201 architecture.

### EfficientNet Applications

- [`application_efficientnet_b0()`](https://keras3.posit.co/dev/reference/application_efficientnet_b0.md)
  : Instantiates the EfficientNetB0 architecture.
- [`application_efficientnet_b1()`](https://keras3.posit.co/dev/reference/application_efficientnet_b1.md)
  : Instantiates the EfficientNetB1 architecture.
- [`application_efficientnet_b2()`](https://keras3.posit.co/dev/reference/application_efficientnet_b2.md)
  : Instantiates the EfficientNetB2 architecture.
- [`application_efficientnet_b3()`](https://keras3.posit.co/dev/reference/application_efficientnet_b3.md)
  : Instantiates the EfficientNetB3 architecture.
- [`application_efficientnet_b4()`](https://keras3.posit.co/dev/reference/application_efficientnet_b4.md)
  : Instantiates the EfficientNetB4 architecture.
- [`application_efficientnet_b5()`](https://keras3.posit.co/dev/reference/application_efficientnet_b5.md)
  : Instantiates the EfficientNetB5 architecture.
- [`application_efficientnet_b6()`](https://keras3.posit.co/dev/reference/application_efficientnet_b6.md)
  : Instantiates the EfficientNetB6 architecture.
- [`application_efficientnet_b7()`](https://keras3.posit.co/dev/reference/application_efficientnet_b7.md)
  : Instantiates the EfficientNetB7 architecture.
- [`application_efficientnet_v2b0()`](https://keras3.posit.co/dev/reference/application_efficientnet_v2b0.md)
  : Instantiates the EfficientNetV2B0 architecture.
- [`application_efficientnet_v2b1()`](https://keras3.posit.co/dev/reference/application_efficientnet_v2b1.md)
  : Instantiates the EfficientNetV2B1 architecture.
- [`application_efficientnet_v2b2()`](https://keras3.posit.co/dev/reference/application_efficientnet_v2b2.md)
  : Instantiates the EfficientNetV2B2 architecture.
- [`application_efficientnet_v2b3()`](https://keras3.posit.co/dev/reference/application_efficientnet_v2b3.md)
  : Instantiates the EfficientNetV2B3 architecture.
- [`application_efficientnet_v2l()`](https://keras3.posit.co/dev/reference/application_efficientnet_v2l.md)
  : Instantiates the EfficientNetV2L architecture.
- [`application_efficientnet_v2m()`](https://keras3.posit.co/dev/reference/application_efficientnet_v2m.md)
  : Instantiates the EfficientNetV2M architecture.
- [`application_efficientnet_v2s()`](https://keras3.posit.co/dev/reference/application_efficientnet_v2s.md)
  : Instantiates the EfficientNetV2S architecture.

### Inception Applications

- [`application_inception_resnet_v2()`](https://keras3.posit.co/dev/reference/application_inception_resnet_v2.md)
  : Instantiates the Inception-ResNet v2 architecture.
- [`application_inception_v3()`](https://keras3.posit.co/dev/reference/application_inception_v3.md)
  : Instantiates the Inception v3 architecture.

### MobileNet Applications

- [`application_mobilenet()`](https://keras3.posit.co/dev/reference/application_mobilenet.md)
  : Instantiates the MobileNet architecture.
- [`application_mobilenet_v2()`](https://keras3.posit.co/dev/reference/application_mobilenet_v2.md)
  : Instantiates the MobileNetV2 architecture.
- [`application_mobilenet_v3_large()`](https://keras3.posit.co/dev/reference/application_mobilenet_v3_large.md)
  : Instantiates the MobileNetV3Large architecture.
- [`application_mobilenet_v3_small()`](https://keras3.posit.co/dev/reference/application_mobilenet_v3_small.md)
  : Instantiates the MobileNetV3Small architecture.

### NASNet Applications

- [`application_nasnet_large()`](https://keras3.posit.co/dev/reference/application_nasnet_large.md)
  : Instantiates a NASNet model in ImageNet mode.
- [`application_nasnet_mobile()`](https://keras3.posit.co/dev/reference/application_nasnet_mobile.md)
  : Instantiates a Mobile NASNet model in ImageNet mode.

### ResNet Applications

- [`application_resnet101()`](https://keras3.posit.co/dev/reference/application_resnet101.md)
  : Instantiates the ResNet101 architecture.
- [`application_resnet101_v2()`](https://keras3.posit.co/dev/reference/application_resnet101_v2.md)
  : Instantiates the ResNet101V2 architecture.
- [`application_resnet152()`](https://keras3.posit.co/dev/reference/application_resnet152.md)
  : Instantiates the ResNet152 architecture.
- [`application_resnet152_v2()`](https://keras3.posit.co/dev/reference/application_resnet152_v2.md)
  : Instantiates the ResNet152V2 architecture.
- [`application_resnet50()`](https://keras3.posit.co/dev/reference/application_resnet50.md)
  : Instantiates the ResNet50 architecture.
- [`application_resnet50_v2()`](https://keras3.posit.co/dev/reference/application_resnet50_v2.md)
  : Instantiates the ResNet50V2 architecture.

### VGG Applications

- [`application_vgg16()`](https://keras3.posit.co/dev/reference/application_vgg16.md)
  : Instantiates the VGG16 model.
- [`application_vgg19()`](https://keras3.posit.co/dev/reference/application_vgg19.md)
  : Instantiates the VGG19 model.

### Xception Applications

- [`application_xception()`](https://keras3.posit.co/dev/reference/application_xception.md)
  : Instantiates the Xception architecture.
