# One-stop utility for preprocessing and encoding structured data.

**Available feature types:**

Note that all features can be referred to by their string name, e.g.
`"integer_categorical"`. When using the string name, the default
argument values are used.

    # Plain float values.
    feature_float(name = NULL)

    # Float values to be preprocessed via featurewise standardization
    # (i.e. via a `layer_normalization()` layer).
    feature_float_normalized(name = NULL)

    # Float values to be preprocessed via linear rescaling
    # (i.e. via a `layer_rescaling` layer).
    feature_float_rescaled(scale = 1., offset = 0., name = NULL)

    # Float values to be discretized. By default, the discrete
    # representation will then be one-hot encoded.
    feature_float_discretized(
      num_bins,
      bin_boundaries = NULL,
      output_mode = "one_hot",
      name = NULL
    )

    # Integer values to be indexed. By default, the discrete
    # representation will then be one-hot encoded.
    feature_integer_categorical(
      max_tokens = NULL,
      num_oov_indices = 1,
      output_mode = "one_hot",
      name = NULL
    )

    # String values to be indexed. By default, the discrete
    # representation will then be one-hot encoded.
    feature_string_categorical(
      max_tokens = NULL,
      num_oov_indices = 1,
      output_mode = "one_hot",
      name = NULL
    )

    # Integer values to be hashed into a fixed number of bins.
    # By default, the discrete representation will then be one-hot encoded.
    feature_integer_hashed(num_bins, output_mode = "one_hot", name = NULL)

    # String values to be hashed into a fixed number of bins.
    # By default, the discrete representation will then be one-hot encoded.
    feature_string_hashed(num_bins, output_mode = "one_hot", name = NULL)

## Usage

``` r
layer_feature_space(
  object,
  features,
  output_mode = "concat",
  crosses = NULL,
  crossing_dim = 32L,
  hashing_dim = 32L,
  num_discretization_bins = 32L,
  name = NULL,
  feature_names = NULL
)

feature_cross(feature_names, crossing_dim, output_mode = "one_hot")

feature_custom(dtype, preprocessor, output_mode)

feature_float(name = NULL)

feature_float_rescaled(scale = 1, offset = 0, name = NULL)

feature_float_normalized(name = NULL)

feature_float_discretized(
  num_bins,
  bin_boundaries = NULL,
  output_mode = "one_hot",
  name = NULL
)

feature_integer_categorical(
  max_tokens = NULL,
  num_oov_indices = 1,
  output_mode = "one_hot",
  name = NULL
)

feature_string_categorical(
  max_tokens = NULL,
  num_oov_indices = 1,
  output_mode = "one_hot",
  name = NULL
)

feature_string_hashed(num_bins, output_mode = "one_hot", name = NULL)

feature_integer_hashed(num_bins, output_mode = "one_hot", name = NULL)
```

## Arguments

- object:

  see description

- features:

  see description

- output_mode:

  A string.

  - For `layer_feature_space()`, one of `"concat"` or `"dict"`. In
    concat mode, all features get concatenated together into a single
    vector. In dict mode, the `FeatureSpace` returns a named list of
    individually encoded features (with the same names as the input list
    names).

  - For the `feature_*` functions, one of: `"int"` `"one_hot"` or
    `"float"`.

- crosses:

  List of features to be crossed together, e.g.
  `crosses=list(c("feature_1", "feature_2"))`. The features will be
  "crossed" by hashing their combined value into a fixed-length vector.

- crossing_dim:

  Default vector size for hashing crossed features. Defaults to `32`.

- hashing_dim:

  Default vector size for hashing features of type `"integer_hashed"`
  and `"string_hashed"`. Defaults to `32`.

- num_discretization_bins:

  Default number of bins to be used for discretizing features of type
  `"float_discretized"`. Defaults to `32`.

- name:

  String, name for the object

- feature_names:

  Named list mapping the names of your features to their type
  specification, e.g. `list(my_feature = "integer_categorical")` or
  `list(my_feature = feature_integer_categorical())`. For a complete
  list of all supported types, see "Available feature types" paragraph
  below.

- dtype:

  string, the output dtype of the feature. E.g., "float32".

- preprocessor:

  A callable.

- scale, offset:

  Passed on to
  [`layer_rescaling()`](https://keras3.posit.co/reference/layer_rescaling.md)

- num_bins, bin_boundaries:

  Passed on to
  [`layer_discretization()`](https://keras3.posit.co/reference/layer_discretization.md)

- max_tokens, num_oov_indices:

  Passed on to
  [`layer_integer_lookup()`](https://keras3.posit.co/reference/layer_integer_lookup.md)
  by `feature_integer_categorical()` or to
  [`layer_string_lookup()`](https://keras3.posit.co/reference/layer_string_lookup.md)
  by `feature_string_categorical()`.

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

**Basic usage with a named list of input data:**

    raw_data <- list(
      float_values = c(0.0, 0.1, 0.2, 0.3),
      string_values = c("zero", "one", "two", "three"),
      int_values = as.integer(c(0, 1, 2, 3))
    )

    dataset <- tfdatasets::tensor_slices_dataset(raw_data)

    feature_space <- layer_feature_space(
      features = list(
        float_values = "float_normalized",
        string_values = "string_categorical",
        int_values = "integer_categorical"
      ),
      crosses = list(c("string_values", "int_values")),
      output_mode = "concat"
    )

    # Before you start using the feature_space(),
    # you must `adapt()` it on some data.
    feature_space |> adapt(dataset)

    # You can call the feature_space() on a named list of
    # data (batched or unbatched).
    output_vector <- feature_space(raw_data)

**Basic usage with `tf.data`:**

    library(tfdatasets)
    # Unlabeled data
    preprocessed_ds <- unlabeled_dataset |>
      dataset_map(feature_space)

    # Labeled data
    preprocessed_ds <- labeled_dataset |>
      dataset_map(function(x, y) tuple(feature_space(x), y))

**Basic usage with the Keras Functional API:**

    # Retrieve a named list of Keras layer_input() objects
    (inputs <- feature_space$get_inputs())

    ## $float_values
    ## <KerasTensor shape=(None, 1), dtype=float32, sparse=False, ragged=False, name=float_values>
    ##
    ## $string_values
    ## <KerasTensor shape=(None, 1), dtype=string, sparse=False, ragged=False, name=string_values>
    ##
    ## $int_values
    ## <KerasTensor shape=(None, 1), dtype=int32, sparse=False, ragged=False, name=int_values>

    # Retrieve the corresponding encoded Keras tensors
    (encoded_features <- feature_space$get_encoded_features())

    ## <KerasTensor shape=(None, 43), dtype=float32, sparse=False, ragged=False, name=keras_tensor_7>

    # Build a Functional model
    outputs <- encoded_features |> layer_dense(1, activation = "sigmoid")
    model <- keras_model(inputs, outputs)

**Customizing each feature or feature cross:**

    feature_space <- layer_feature_space(
      features = list(
        float_values = feature_float_normalized(),
        string_values = feature_string_categorical(max_tokens = 10),
        int_values = feature_integer_categorical(max_tokens = 10)
      ),
      crosses = list(
        feature_cross(c("string_values", "int_values"), crossing_dim = 32)
      ),
      output_mode = "concat"
    )

**Returning a dict (a named list) of integer-encoded features:**

    feature_space <- layer_feature_space(
      features = list(
        "string_values" = feature_string_categorical(output_mode = "int"),
        "int_values" = feature_integer_categorical(output_mode = "int")
      ),
      crosses = list(
        feature_cross(
          feature_names = c("string_values", "int_values"),
          crossing_dim = 32,
          output_mode = "int"
        )
      ),
      output_mode = "dict"
    )

**Specifying your own Keras preprocessing layer:**

    # Let's say that one of the features is a short text paragraph that
    # we want to encode as a vector (one vector per paragraph) via TF-IDF.
    data <- list(text = c("1st string", "2nd string", "3rd string"))

    # There's a Keras layer for this: layer_text_vectorization()
    custom_layer <- layer_text_vectorization(output_mode = "tf_idf")

    # We can use feature_custom() to create a custom feature
    # that will use our preprocessing layer.
    feature_space <- layer_feature_space(
      features = list(
        text = feature_custom(preprocessor = custom_layer,
                              dtype = "string",
                              output_mode = "float"
        )
      ),
      output_mode = "concat"
    )
    feature_space |> adapt(tfdatasets::tensor_slices_dataset(data))
    output_vector <- feature_space(data)

**Retrieving the underlying Keras preprocessing layers:**

    # The preprocessing layer of each feature is available in `$preprocessors`.
    preprocessing_layer <- feature_space$preprocessors$feature1

    # The crossing layer of each feature cross is available in `$crossers`.
    # It's an instance of layer_hashed_crossing()
    crossing_layer <- feature_space$crossers[["feature1_X_feature2"]]

**Saving and reloading a FeatureSpace:**

    feature_space$save("featurespace.keras")
    reloaded_feature_space <- keras$models$load_model("featurespace.keras")

## See also

- <https://keras.io/api/utils/feature_space#featurespace-class>

Other preprocessing layers:  
[`layer_aug_mix()`](https://keras3.posit.co/reference/layer_aug_mix.md)  
[`layer_auto_contrast()`](https://keras3.posit.co/reference/layer_auto_contrast.md)  
[`layer_category_encoding()`](https://keras3.posit.co/reference/layer_category_encoding.md)  
[`layer_center_crop()`](https://keras3.posit.co/reference/layer_center_crop.md)  
[`layer_cut_mix()`](https://keras3.posit.co/reference/layer_cut_mix.md)  
[`layer_discretization()`](https://keras3.posit.co/reference/layer_discretization.md)  
[`layer_equalization()`](https://keras3.posit.co/reference/layer_equalization.md)  
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

Other utils:  
[`audio_dataset_from_directory()`](https://keras3.posit.co/reference/audio_dataset_from_directory.md)  
[`clear_session()`](https://keras3.posit.co/reference/clear_session.md)  
[`config_disable_interactive_logging()`](https://keras3.posit.co/reference/config_disable_interactive_logging.md)  
[`config_disable_traceback_filtering()`](https://keras3.posit.co/reference/config_disable_traceback_filtering.md)  
[`config_enable_interactive_logging()`](https://keras3.posit.co/reference/config_enable_interactive_logging.md)  
[`config_enable_traceback_filtering()`](https://keras3.posit.co/reference/config_enable_traceback_filtering.md)  
[`config_is_interactive_logging_enabled()`](https://keras3.posit.co/reference/config_is_interactive_logging_enabled.md)  
[`config_is_traceback_filtering_enabled()`](https://keras3.posit.co/reference/config_is_traceback_filtering_enabled.md)  
[`get_file()`](https://keras3.posit.co/reference/get_file.md)  
[`get_source_inputs()`](https://keras3.posit.co/reference/get_source_inputs.md)  
[`image_array_save()`](https://keras3.posit.co/reference/image_array_save.md)  
[`image_dataset_from_directory()`](https://keras3.posit.co/reference/image_dataset_from_directory.md)  
[`image_from_array()`](https://keras3.posit.co/reference/image_from_array.md)  
[`image_load()`](https://keras3.posit.co/reference/image_load.md)  
[`image_smart_resize()`](https://keras3.posit.co/reference/image_smart_resize.md)  
[`image_to_array()`](https://keras3.posit.co/reference/image_to_array.md)  
[`normalize()`](https://keras3.posit.co/reference/normalize.md)  
[`pad_sequences()`](https://keras3.posit.co/reference/pad_sequences.md)  
[`set_random_seed()`](https://keras3.posit.co/reference/set_random_seed.md)  
[`split_dataset()`](https://keras3.posit.co/reference/split_dataset.md)  
[`text_dataset_from_directory()`](https://keras3.posit.co/reference/text_dataset_from_directory.md)  
[`timeseries_dataset_from_array()`](https://keras3.posit.co/reference/timeseries_dataset_from_array.md)  
[`to_categorical()`](https://keras3.posit.co/reference/to_categorical.md)  
[`zip_lists()`](https://keras3.posit.co/reference/zip_lists.md)  
