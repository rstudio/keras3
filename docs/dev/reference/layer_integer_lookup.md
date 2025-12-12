# A preprocessing layer that maps integers to (possibly encoded) indices.

This layer maps a set of arbitrary integer input tokens into indexed
integer output via a table-based vocabulary lookup. The layer's output
indices will be contiguously arranged up to the maximum vocab size, even
if the input tokens are non-continguous or unbounded. The layer supports
multiple options for encoding the output via `output_mode`, and has
optional support for out-of-vocabulary (OOV) tokens and masking.

The vocabulary for the layer must be either supplied on construction or
learned via [`adapt()`](https://keras3.posit.co/dev/reference/adapt.md).
During [`adapt()`](https://keras3.posit.co/dev/reference/adapt.md), the
layer will analyze a data set, determine the frequency of individual
integer tokens, and create a vocabulary from them. If the vocabulary is
capped in size, the most frequent tokens will be used to create the
vocabulary and all others will be treated as OOV.

There are two possible output modes for the layer. When `output_mode` is
`"int"`, input integers are converted to their index in the vocabulary
(an integer). When `output_mode` is `"multi_hot"`, `"count"`, or
`"tf_idf"`, input integers are encoded into an array where each
dimension corresponds to an element in the vocabulary.

The vocabulary can optionally contain a mask token as well as an OOV
token (which can optionally occupy multiple indices in the vocabulary,
as set by `num_oov_indices`). The position of these tokens in the
vocabulary is fixed. When `output_mode` is `"int"`, the vocabulary will
begin with the mask token at index 0, followed by OOV indices, followed
by the rest of the vocabulary. When `output_mode` is `"multi_hot"`,
`"count"`, or `"tf_idf"` the vocabulary will begin with OOV indices and
instances of the mask token will be dropped.

**Note:** This layer uses TensorFlow internally. It cannot be used as
part of the compiled computation graph of a model with any backend other
than TensorFlow. It can however be used with any backend when running
eagerly. It can also always be used as part of an input preprocessing
pipeline with any backend (outside the model itself), which is how we
recommend to use this layer.

**Note:** This layer is safe to use inside a `tf.data` pipeline
(independently of which backend you're using).

**Note:** If working with layer outputs directly (e.g., not passing
outputs to another layer, but using them in lower-level operations like
`[` and `op_*`): the returned indices are 0-based. However, with default
settings, the first (`0`) index is the OOV token, so the returned
indices are offset by `1` and may appear to be 1-based.

## Usage

``` r
layer_integer_lookup(
  object,
  max_tokens = NULL,
  num_oov_indices = 1L,
  mask_token = NULL,
  oov_token = -1L,
  vocabulary = NULL,
  vocabulary_dtype = "int64",
  idf_weights = NULL,
  invert = FALSE,
  output_mode = "int",
  sparse = FALSE,
  pad_to_max_tokens = FALSE,
  name = NULL,
  ...
)
```

## Arguments

- object:

  Object to compose the layer with. A tensor, array, or sequential
  model.

- max_tokens:

  Maximum size of the vocabulary for this layer. This should only be
  specified when adapting the vocabulary or when setting
  `pad_to_max_tokens=TRUE`. If NULL, there is no cap on the size of the
  vocabulary. Note that this size includes the OOV and mask tokens.
  Defaults to `NULL`.

- num_oov_indices:

  The number of out-of-vocabulary tokens to use. If this value is more
  than 1, OOV inputs are modulated to determine their OOV value. If this
  value is 0, OOV inputs will cause an error when calling the layer.
  Defaults to `1`.

- mask_token:

  An integer token that represents masked inputs. When `output_mode` is
  `"int"`, the token is included in vocabulary and mapped to index 0. In
  other output modes, the token will not appear in the vocabulary and
  instances of the mask token in the input will be dropped. If set to
  NULL, no mask term will be added. Defaults to `NULL`.

- oov_token:

  Only used when `invert` is `TRUE`. The token to return for OOV
  indices. Defaults to `-1`.

- vocabulary:

  Optional. Either an array of integers or a string path to a text file.
  If passing an array, can pass a list, list, 1D NumPy array, or 1D
  tensor containing the integer vocabulary terms. If passing a file
  path, the file should contain one line per term in the vocabulary. If
  this argument is set, there is no need to
  [`adapt()`](https://keras3.posit.co/dev/reference/adapt.md) the layer.

- vocabulary_dtype:

  The dtype of the vocabulary terms, for example `"int64"` or `"int32"`.
  Defaults to `"int64"`.

- idf_weights:

  Only valid when `output_mode` is `"tf_idf"`. A list, list, 1D NumPy
  array, or 1D tensor or the same length as the vocabulary, containing
  the floating point inverse document frequency weights, which will be
  multiplied by per sample term counts for the final TF-IDF weight. If
  the `vocabulary` argument is set, and `output_mode` is `"tf_idf"`,
  this argument must be supplied.

- invert:

  Only valid when `output_mode` is `"int"`. If `TRUE`, this layer will
  map indices to vocabulary items instead of mapping vocabulary items to
  indices. Defaults to `FALSE`.

- output_mode:

  Specification for the output of the layer. Values can be `"int"`,
  `"one_hot"`, `"multi_hot"`, `"count"`, or `"tf_idf"` configuring the
  layer as follows:

  - `"int"`: Return the vocabulary indices of the input tokens.

  - `"one_hot"`: Encodes each individual element in the input into an
    array the same size as the vocabulary, containing a 1 at the element
    index. If the last dimension is size 1, will encode on that
    dimension. If the last dimension is not size 1, will append a new
    dimension for the encoded output.

  - `"multi_hot"`: Encodes each sample in the input into a single array
    the same size as the vocabulary, containing a 1 for each vocabulary
    term present in the sample. Treats the last dimension as the sample
    dimension, if input shape is `(..., sample_length)`, output shape
    will be `(..., num_tokens)`.

  - `"count"`: As `"multi_hot"`, but the int array contains a count of
    the number of times the token at that index appeared in the sample.

  - `"tf_idf"`: As `"multi_hot"`, but the TF-IDF algorithm is applied to
    find the value in each token slot. For `"int"` output, any shape of
    input and output is supported. For all other output modes, currently
    only output up to rank 2 is supported. Defaults to `"int"`.

- sparse:

  Boolean. Only applicable to `"multi_hot"`, `"count"`, and `"tf_idf"`
  output modes. Only supported with TensorFlow backend. If `TRUE`,
  returns a `SparseTensor` instead of a dense `Tensor`. Defaults to
  `FALSE`.

- pad_to_max_tokens:

  Only applicable when `output_mode` is `"multi_hot"`, `"count"`, or
  `"tf_idf"`. If `TRUE`, the output will have its feature axis padded to
  `max_tokens` even if the number of unique tokens in the vocabulary is
  less than `max_tokens`, resulting in a tensor of shape
  `(batch_size, max_tokens)` regardless of vocabulary size. Defaults to
  `FALSE`.

- name:

  String, name for the object

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

## Examples

**Creating a lookup layer with a known vocabulary**

This example creates a lookup layer with a pre-existing vocabulary.

    vocab <- c(12, 36, 1138, 42) |> as.integer()
    data <- op_array(rbind(c(12, 1138, 42),
                           c(42, 1000, 36)))  # Note OOV tokens
    out <- data |> layer_integer_lookup(vocabulary = vocab)
    out

    ## tf.Tensor(
    ## [[1 3 4]
    ##  [4 0 2]], shape=(2, 3), dtype=int64)

**Creating a lookup layer with an adapted vocabulary**

This example creates a lookup layer and generates the vocabulary by
analyzing the dataset.

    data <- op_array(rbind(c(12, 1138, 42),
                           c(42, 1000, 36)))  # Note OOV tokens
    layer <- layer_integer_lookup()
    layer |> adapt(data)
    layer |> get_vocabulary() |> str()

    ## List of 6
    ##  $ : int -1
    ##  $ : num 42
    ##  $ : num 1138
    ##  $ : num 1000
    ##  $ : num 36
    ##  $ : num 12

Note that the OOV token -1 have been added to the vocabulary. The
remaining tokens are sorted by frequency (42, which has 2 occurrences,
is first) then by inverse sort order.

    layer(data)

    ## tf.Tensor(
    ## [[5 2 1]
    ##  [1 3 4]], shape=(2, 3), dtype=int64)

**Lookups with multiple OOV indices**

This example demonstrates how to use a lookup layer with multiple OOV
indices. When a layer is created with more than one OOV index, any OOV
tokens are hashed into the number of OOV buckets, distributing OOV
tokens in a deterministic fashion across the set.

    vocab <- c(12, 36, 1138, 42) |> as.integer()
    data <- op_array(rbind(c(12, 1138, 42),
                           c(37, 1000, 36)))  # Note OOV tokens
    out <- data |>
      layer_integer_lookup(vocabulary = vocab,
                           num_oov_indices = 2)
    out

    ## tf.Tensor(
    ## [[2 4 5]
    ##  [1 0 3]], shape=(2, 3), dtype=int64)

Note that the output for OOV token 37 is 1, while the output for OOV
token 1000 is 0. The in-vocab terms have their output index increased by
1 from earlier examples (12 maps to 2, etc) in order to make space for
the extra OOV token.

**One-hot output**

Configure the layer with `output_mode='one_hot'`. Note that the first
`num_oov_indices` dimensions in the ont_hot encoding represent OOV
values.

    vocab <- c(12, 36, 1138, 42) |> as.integer()
    data <- op_array(c(12, 36, 1138, 42, 7), 'int32')  # Note OOV tokens
    layer <- layer_integer_lookup(vocabulary = vocab,
                                  output_mode = 'one_hot')
    layer(data)

    ## tf.Tensor(
    ## [[0 1 0 0 0]
    ##  [0 0 1 0 0]
    ##  [0 0 0 1 0]
    ##  [0 0 0 0 1]
    ##  [1 0 0 0 0]], shape=(5, 5), dtype=int64)

**Multi-hot output**

Configure the layer with `output_mode = 'multi_hot'`. Note that the
first `num_oov_indices` dimensions in the multi_hot encoding represent
OOV tokens

    vocab <- c(12, 36, 1138, 42) |> as.integer()
    data <- op_array(rbind(c(12, 1138, 42, 42),
                          c(42,    7, 36,  7)), "int64")  # Note OOV tokens
    layer <- layer_integer_lookup(vocabulary = vocab,
                                  output_mode = 'multi_hot')
    layer(data)

    ## tf.Tensor(
    ## [[0 1 0 1 1]
    ##  [1 0 1 0 1]], shape=(2, 5), dtype=int64)

**Token count output**

Configure the layer with `output_mode='count'`. As with multi_hot
output, the first `num_oov_indices` dimensions in the output represent
OOV tokens.

    vocab <- c(12, 36, 1138, 42) |> as.integer()
    data <- rbind(c(12, 1138, 42, 42),
                  c(42,    7, 36,  7)) |> op_array("int64")
    layer <- layer_integer_lookup(vocabulary = vocab,
                                  output_mode = 'count')
    layer(data)

    ## tf.Tensor(
    ## [[0 1 0 1 2]
    ##  [2 0 1 0 1]], shape=(2, 5), dtype=int64)

**TF-IDF output**

Configure the layer with `output_mode='tf_idf'`. As with multi_hot
output, the first `num_oov_indices` dimensions in the output represent
OOV tokens.

Each token bin will output `token_count * idf_weight`, where the idf
weights are the inverse document frequency weights per token. These
should be provided along with the vocabulary. Note that the `idf_weight`
for OOV tokens will default to the average of all idf weights passed in.

    vocab <- c(12, 36, 1138, 42) |> as.integer()
    idf_weights <- c(0.25, 0.75, 0.6, 0.4)
    data <- rbind(c(12, 1138, 42, 42),
                  c(42,    7, 36,  7)) |> op_array("int64")
    layer <- layer_integer_lookup(output_mode = 'tf_idf',
                                  vocabulary = vocab,
                                  idf_weights = idf_weights)
    layer(data)

    ## tf.Tensor(
    ## [[0.   0.25 0.   0.6  0.8 ]
    ##  [1.   0.   0.75 0.   0.4 ]], shape=(2, 5), dtype=float32)

To specify the idf weights for oov tokens, you will need to pass the
entire vocabulary including the leading oov token.

    vocab <- c(-1, 12, 36, 1138, 42) |> as.integer()
    idf_weights <- c(0.9, 0.25, 0.75, 0.6, 0.4)
    data <- rbind(c(12, 1138, 42, 42),
                  c(42,    7, 36,  7)) |> op_array("int64")
    layer <- layer_integer_lookup(output_mode = 'tf_idf',
                                  vocabulary = vocab,
                                  idf_weights = idf_weights)
    layer(data)

    ## tf.Tensor(
    ## [[0.   0.25 0.   0.6  0.8 ]
    ##  [1.8  0.   0.75 0.   0.4 ]], shape=(2, 5), dtype=float32)

When adapting the layer in `"tf_idf"` mode, each input sample will be
considered a document, and IDF weight per token will be calculated as:
`log(1 + num_documents / (1 + token_document_count))`.

**Inverse lookup**

This example demonstrates how to map indices to tokens using this layer.
(You can also use
[`adapt()`](https://keras3.posit.co/dev/reference/adapt.md) with
`inverse = TRUE`, but for simplicity we'll pass the vocab in this
example.)

    vocab <- c(12, 36, 1138, 42) |> as.integer()
    data <- op_array(c(1, 3, 4,
                       4, 0, 2)) |> op_reshape(c(2,-1)) |> op_cast("int32")
    layer <- layer_integer_lookup(vocabulary = vocab, invert = TRUE)
    layer(data)

    ## tf.Tensor(
    ## [[  12 1138   42]
    ##  [  42   -1   36]], shape=(2, 3), dtype=int64)

Note that the first index correspond to the oov token by default.

**Forward and inverse lookup pairs**

This example demonstrates how to use the vocabulary of a standard lookup
layer to create an inverse lookup layer.

    vocab <- c(12, 36, 1138, 42) |> as.integer()
    data <- op_array(rbind(c(12, 1138, 42), c(42, 1000, 36)), "int32")
    layer <- layer_integer_lookup(vocabulary = vocab)
    i_layer <- layer_integer_lookup(vocabulary = get_vocabulary(layer),
                                    invert = TRUE)
    int_data <- layer(data)
    i_layer(int_data)

    ## tf.Tensor(
    ## [[  12 1138   42]
    ##  [  42   -1   36]], shape=(2, 3), dtype=int64)

In this example, the input token 1000 resulted in an output of -1, since
1000 was not in the vocabulary - it got represented as an OOV, and all
OOV tokens are returned as -1 in the inverse layer. Also, note that for
the inverse to work, you must have already set the forward layer
vocabulary either directly or via
[`adapt()`](https://keras3.posit.co/dev/reference/adapt.md) before
calling
[`get_vocabulary()`](https://keras3.posit.co/dev/reference/layer_text_vectorization.md).

## See also

- <https://keras.io/api/layers/preprocessing_layers/categorical/integer_lookup#integerlookup-class>

Other categorical features preprocessing layers:  
[`layer_category_encoding()`](https://keras3.posit.co/dev/reference/layer_category_encoding.md)  
[`layer_hashed_crossing()`](https://keras3.posit.co/dev/reference/layer_hashed_crossing.md)  
[`layer_hashing()`](https://keras3.posit.co/dev/reference/layer_hashing.md)  
[`layer_string_lookup()`](https://keras3.posit.co/dev/reference/layer_string_lookup.md)  

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
[`layer_text_vectorization()`](https://keras3.posit.co/dev/reference/layer_text_vectorization.md)  

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
