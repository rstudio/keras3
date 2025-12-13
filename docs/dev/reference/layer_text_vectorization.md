# A preprocessing layer which maps text features to integer sequences.

This layer has basic options for managing text in a Keras model. It
transforms a batch of strings (one example = one string) into either a
list of token indices (one example = 1D tensor of integer token indices)
or a dense representation (one example = 1D tensor of float values
representing data about the example's tokens). This layer is meant to
handle natural language inputs. To handle simple string inputs
(categorical strings or pre-tokenized strings) see
[`layer_string_lookup()`](https://keras3.posit.co/dev/reference/layer_string_lookup.md).

The vocabulary for the layer must be either supplied on construction or
learned via [`adapt()`](https://keras3.posit.co/dev/reference/adapt.md).
When this layer is adapted, it will analyze the dataset, determine the
frequency of individual string values, and create a vocabulary from
them. This vocabulary can have unlimited size or be capped, depending on
the configuration options for this layer; if there are more unique
values in the input than the maximum vocabulary size, the most frequent
terms will be used to create the vocabulary.

The processing of each example contains the following steps:

1.  Standardize each example (usually lowercasing + punctuation
    stripping)

2.  Split each example into substrings (usually words)

3.  Recombine substrings into tokens (usually ngrams)

4.  Index tokens (associate a unique int value with each token)

5.  Transform each example using this index, either into a vector of
    ints or a dense float vector.

Some notes on passing callables to customize splitting and normalization
for this layer:

1.  Any callable can be passed to this Layer, but if you want to
    serialize this object you should only pass functions that are
    registered Keras serializables (see
    [`register_keras_serializable()`](https://keras3.posit.co/dev/reference/register_keras_serializable.md)
    for more details).

2.  When using a custom callable for `standardize`, the data received by
    the callable will be exactly as passed to this layer. The callable
    should return a tensor of the same shape as the input.

3.  When using a custom callable for `split`, the data received by the
    callable will have the 1st dimension squeezed out - instead of
    `list("string to split", "another string to split")`, the Callable
    will see `c("string to split", "another string to split")`. The
    callable should return a `tf.Tensor` of dtype `string` with the
    first dimension containing the split tokens - in this example, we
    should see something like
    `list(c("string", "to", "split"), c("another", "string", "to", "split"))`.

**Note:** This layer uses TensorFlow internally. It cannot be used as
part of the compiled computation graph of a model with any backend other
than TensorFlow. It can however be used with any backend when running
eagerly. It can also always be used as part of an input preprocessing
pipeline with any backend (outside the model itself), which is how we
recommend to use this layer.

**Note:** This layer is safe to use inside a `tf.data` pipeline
(independently of which backend you're using).

## Usage

``` r
layer_text_vectorization(
  object,
  max_tokens = NULL,
  standardize = "lower_and_strip_punctuation",
  split = "whitespace",
  ngrams = NULL,
  output_mode = "int",
  output_sequence_length = NULL,
  pad_to_max_tokens = FALSE,
  vocabulary = NULL,
  idf_weights = NULL,
  sparse = FALSE,
  ragged = FALSE,
  encoding = "utf-8",
  name = NULL,
  ...
)

get_vocabulary(object, include_special_tokens = TRUE)

set_vocabulary(object, vocabulary, idf_weights = NULL, ...)
```

## Arguments

- object:

  Object to compose the layer with. A tensor, array, or sequential
  model.

- max_tokens:

  Maximum size of the vocabulary for this layer. This should only be
  specified when adapting a vocabulary or when setting
  `pad_to_max_tokens=TRUE`. Note that this vocabulary contains 1 OOV
  token, so the effective number of tokens is
  `(max_tokens - 1 - (1 if output_mode == "int" else 0))`.

- standardize:

  Optional specification for standardization to apply to the input text.
  Values can be:

  - `NULL`: No standardization.

  - `"lower_and_strip_punctuation"`: Text will be lowercased and all
    punctuation removed.

  - `"lower"`: Text will be lowercased.

  - `"strip_punctuation"`: All punctuation will be removed.

  - Callable: Inputs will passed to the callable function, which should
    be standardized and returned.

- split:

  Optional specification for splitting the input text. Values can be:

  - `NULL`: No splitting.

  - `"whitespace"`: Split on whitespace.

  - `"character"`: Split on each unicode character.

  - Callable: Standardized inputs will passed to the callable function,
    which should be split and returned.

- ngrams:

  Optional specification for ngrams to create from the possibly-split
  input text. Values can be `NULL`, an integer or list of integers;
  passing an integer will create ngrams up to that integer, and passing
  a list of integers will create ngrams for the specified values in the
  list. Passing `NULL` means that no ngrams will be created.

- output_mode:

  Optional specification for the output of the layer. Values can be
  `"int"`, `"multi_hot"`, `"count"` or `"tf_idf"`, configuring the layer
  as follows:

  - `"int"`: Outputs integer indices, one integer index per split string
    token. When `output_mode == "int"`, 0 is reserved for masked
    locations; this reduces the vocab size to `max_tokens - 2` instead
    of `max_tokens - 1`.

  - `"multi_hot"`: Outputs a single int array per batch, of either
    vocab_size or max_tokens size, containing 1s in all elements where
    the token mapped to that index exists at least once in the batch
    item.

  - `"count"`: Like `"multi_hot"`, but the int array contains a count of
    the number of times the token at that index appeared in the batch
    item.

  - `"tf_idf"`: Like `"multi_hot"`, but the TF-IDF algorithm is applied
    to find the value in each token slot. For `"int"` output, any shape
    of input and output is supported. For all other output modes,
    currently only rank 1 inputs (and rank 2 outputs after splitting)
    are supported.

- output_sequence_length:

  Only valid in INT mode. If set, the output will have its time
  dimension padded or truncated to exactly `output_sequence_length`
  values, resulting in a tensor of shape
  `(batch_size, output_sequence_length)` regardless of how many tokens
  resulted from the splitting step. Defaults to `NULL`. If `ragged` is
  `TRUE` then `output_sequence_length` may still truncate the output.

- pad_to_max_tokens:

  Only valid in `"multi_hot"`, `"count"`, and `"tf_idf"` modes. If
  `TRUE`, the output will have its feature axis padded to `max_tokens`
  even if the number of unique tokens in the vocabulary is less than
  `max_tokens`, resulting in a tensor of shape
  `(batch_size, max_tokens)` regardless of vocabulary size. Defaults to
  `FALSE`.

- vocabulary:

  Optional. Either an array of strings or a string path to a text file.
  If passing an array, can pass a list, list, 1D NumPy array, or 1D
  tensor containing the string vocabulary terms. If passing a file path,
  the file should contain one line per term in the vocabulary. If this
  argument is set, there is no need to
  [`adapt()`](https://keras3.posit.co/dev/reference/adapt.md) the layer.

- idf_weights:

  An R vector, 1D numpy array, or 1D tensor of inverse document
  frequency weights with equal length to vocabulary. Must be set if
  output_mode is "tf_idf". Should not be set otherwise.

- sparse:

  Boolean. Only applicable to `"multi_hot"`, `"count"`, and `"tf_idf"`
  output modes. Only supported with TensorFlow backend. If `TRUE`,
  returns a `SparseTensor` instead of a dense `Tensor`. Defaults to
  `FALSE`.

- ragged:

  Boolean. Only applicable to `"int"` output mode. Only supported with
  TensorFlow backend. If `TRUE`, returns a `RaggedTensor` instead of a
  dense `Tensor`, where each sequence may have a different length after
  string splitting. Defaults to `FALSE`.

- encoding:

  Optional. The text encoding to use to interpret the input strings.
  Defaults to `"utf-8"`.

- name:

  String, name for the object

- ...:

  For forward/backward compatability.

- include_special_tokens:

  If TRUE, the returned vocabulary will include the padding and OOV
  tokens, and a term's index in the vocabulary will equal the term's
  index when calling the layer. If FALSE, the returned vocabulary will
  not include any padding or OOV tokens.

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

## Examples

This example instantiates a `TextVectorization` layer that lowercases
text, splits on whitespace, strips punctuation, and outputs integer
vocab indices.

    max_tokens <- 5000  # Maximum vocab size.
    max_len <- 4  # Sequence length to pad the outputs to.
    # Create the layer.
    vectorize_layer <- layer_text_vectorization(
        max_tokens = max_tokens,
        output_mode = 'int',
        output_sequence_length = max_len)

    # Now that the vocab layer has been created, call `adapt` on the
    # list of strings to create the vocabulary.
    vectorize_layer %>% adapt(c("foo bar", "bar baz", "baz bada boom"))

    # Now, the layer can map strings to integers -- you can use an
    # embedding layer to map these integers to learned embeddings.
    input_data <- rbind("foo qux bar", "qux baz")
    vectorize_layer(input_data)

    ## tf.Tensor(
    ## [[4 1 3 0]
    ##  [1 2 0 0]], shape=(2, 4), dtype=int64)

This example instantiates a `TextVectorization` layer by passing a list
of vocabulary terms to the layer's `initialize()` method.

    vocab_data <- c("earth", "wind", "and", "fire")
    max_len <- 4  # Sequence length to pad the outputs to.
    # Create the layer, passing the vocab directly. You can also pass the
    # vocabulary arg a path to a file containing one vocabulary word per
    # line.
    vectorize_layer <- layer_text_vectorization(
        max_tokens = max_tokens,
        output_mode = 'int',
        output_sequence_length = max_len,
        vocabulary = vocab_data)

    # Because we've passed the vocabulary directly, we don't need to adapt
    # the layer - the vocabulary is already set. The vocabulary contains the
    # padding token ('') and OOV token ('[UNK]')
    # as well as the passed tokens.
    vectorize_layer %>% get_vocabulary()

    ## [1] ""      "[UNK]" "earth" "wind"  "and"   "fire"

    # ['', '[UNK]', 'earth', 'wind', 'and', 'fire']

## See also

- <https://keras.io/api/layers/preprocessing_layers/text/text_vectorization#textvectorization-class>

Other preprocessing layers:  
[`layer_aug_mix()`](https://keras3.posit.co/dev/reference/layer_aug_mix.md)  
[`layer_auto_contrast()`](https://keras3.posit.co/dev/reference/layer_auto_contrast.md)  
[`layer_category_encoding()`](https://keras3.posit.co/dev/reference/layer_category_encoding.md)  
[`layer_center_crop()`](https://keras3.posit.co/dev/reference/layer_center_crop.md)  
[`layer_cut_mix()`](https://keras3.posit.co/dev/reference/layer_cut_mix.md)  
[`layer_discretization()`](https://keras3.posit.co/dev/reference/layer_discretization.md)  
[`layer_equalization()`](https://keras3.posit.co/dev/reference/layer_equalization.md)  
[`layer_feature_space()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)  
[`layer_hashed_crossing()`](https://keras3.posit.co/dev/reference/layer_hashed_crossing.md)  
[`layer_hashing()`](https://keras3.posit.co/dev/reference/layer_hashing.md)  
[`layer_integer_lookup()`](https://keras3.posit.co/dev/reference/layer_integer_lookup.md)  
[`layer_max_num_bounding_boxes()`](https://keras3.posit.co/dev/reference/layer_max_num_bounding_boxes.md)  
[`layer_mel_spectrogram()`](https://keras3.posit.co/dev/reference/layer_mel_spectrogram.md)  
[`layer_mix_up()`](https://keras3.posit.co/dev/reference/layer_mix_up.md)  
[`layer_normalization()`](https://keras3.posit.co/dev/reference/layer_normalization.md)  
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
[`layer_rescaling()`](https://keras3.posit.co/dev/reference/layer_rescaling.md)  
[`layer_resizing()`](https://keras3.posit.co/dev/reference/layer_resizing.md)  
[`layer_solarization()`](https://keras3.posit.co/dev/reference/layer_solarization.md)  
[`layer_stft_spectrogram()`](https://keras3.posit.co/dev/reference/layer_stft_spectrogram.md)  
[`layer_string_lookup()`](https://keras3.posit.co/dev/reference/layer_string_lookup.md)  

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
