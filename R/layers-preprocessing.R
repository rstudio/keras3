


#' A preprocessing layer which encodes integer features.
#'
#' @description
#' This layer provides options for condensing data into a categorical encoding
#' when the total number of tokens are known in advance. It accepts integer
#' values as inputs, and it outputs a dense or sparse representation of those
#' inputs. For integer inputs where the total number of tokens is not known,
#' use `layer_integer_lookup()` instead.
#'
#' **Note:** This layer is safe to use inside a `tf.data` pipeline
#' (independently of which backend you're using).
#'
#' # Examples
#' **One-hot encoding data**
#'
#' ```{r}
#' layer <- layer_category_encoding(num_tokens = 4, output_mode = "one_hot")
#' x <- op_array(c(3, 2, 0, 1), "int32")
#' layer(x)
#' ```
#'
#' **Multi-hot encoding data**
#'
#' ```{r}
#' layer <- layer_category_encoding(num_tokens = 4, output_mode = "multi_hot")
#' x <- op_array(rbind(c(0, 1),
#'                    c(0, 0),
#'                    c(1, 2),
#'                    c(3, 1)), "int32")
#' layer(x)
#' ```
#'
#' **Using weighted inputs in `"count"` mode**
#'
#' ```{r, eval = FALSE}
#' layer <- layer_category_encoding(num_tokens = 4, output_mode = "count")
#' count_weights <- op_array(rbind(c(.1, .2),
#'                                c(.1, .1),
#'                                c(.2, .3),
#'                                c(.4, .2)))
#' x <- op_array(rbind(c(0, 1),
#'                    c(0, 0),
#'                    c(1, 2),
#'                    c(3, 1)), "int32")
#' layer(x, count_weights = count_weights)
#' #   array([[01, 02, 0. , 0. ],
#' #          [02, 0. , 0. , 0. ],
#' #          [0. , 02, 03, 0. ],
#' #          [0. , 02, 0. , 04]]>
#' ```
#'
#' # Call Arguments
#' - `inputs`: A 1D or 2D tensor of integer inputs.
#' - `count_weights`: A tensor in the same shape as `inputs` indicating the
#'     weight for each sample value when summing up in `count` mode.
#'     Not used in `"multi_hot"` or `"one_hot"` modes.
#'
#' @param num_tokens
#' The total number of tokens the layer should support. All
#' inputs to the layer must integers in the range `0 <= value <
#' num_tokens`, or an error will be thrown.
#'
#' @param output_mode
#' Specification for the output of the layer.
#' Values can be `"one_hot"`, `"multi_hot"` or `"count"`,
#' configuring the layer as follows:
#'     - `"one_hot"`: Encodes each individual element in the input
#'         into an array of `num_tokens` size, containing a 1 at the
#'         element index. If the last dimension is size 1, will encode
#'         on that dimension. If the last dimension is not size 1,
#'         will append a new dimension for the encoded output.
#'     - `"multi_hot"`: Encodes each sample in the input into a single
#'         array of `num_tokens` size, containing a 1 for each
#'         vocabulary term present in the sample. Treats the last
#'         dimension as the sample dimension, if input shape is
#'         `(..., sample_length)`, output shape will be
#'         `(..., num_tokens)`.
#'     - `"count"`: Like `"multi_hot"`, but the int array contains a
#'         count of the number of times the token at that index
#'         appeared in the sample.
#' For all output modes, currently only output up to rank 2 is
#' supported.
#' Defaults to `"multi_hot"`.
#'
#' @param sparse
#' Whether to return a sparse tensor; for backends that support
#' sparse tensors.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit layer_dense return
#' @export
#' @family categorical features preprocessing layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/categorical/category_encoding#categoryencoding-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/CategoryEncoding>
#' @tether keras.layers.CategoryEncoding
layer_category_encoding <-
function (object, num_tokens = NULL, output_mode = "multi_hot",
    sparse = FALSE, ...)
{
    args <- capture_args(list(output_mode = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape,
        num_tokens = as_integer), ignore = "object")
    create_layer(keras$layers$CategoryEncoding, object, args)
}


#' A preprocessing layer which crops images.
#'
#' @description
#' This layers crops the central portion of the images to a target size. If an
#' image is smaller than the target size, it will be resized and cropped
#' so as to return the largest possible window in the image that matches
#' the target aspect ratio.
#'
#' Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`).
#'
#' # Input Shape
#' 3D (unbatched) or 4D (batched) tensor with shape:
#' `(..., height, width, channels)`, in `"channels_last"` format,
#' or `(..., channels, height, width)`, in `"channels_first"` format.
#'
#' # Output Shape
#' 3D (unbatched) or 4D (batched) tensor with shape:
#'     `(..., target_height, target_width, channels)`,
#'     or `(..., channels, target_height, target_width)`,
#'     in `"channels_first"` format.
#'
#' If the input height/width is even and the target height/width is odd (or
#' inversely), the input image is left-padded by 1 pixel.
#'
#' **Note:** This layer is safe to use inside a `tf.data` pipeline
#' (independently of which backend you're using).
#'
#' @param height
#' Integer, the height of the output shape.
#'
#' @param width
#' Integer, the width of the output shape.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape `(batch, height, width, channels)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, channels, height, width)`. It defaults to the
#' `image_data_format` value found in your Keras config file at
#' `~/.keras/keras.json`. If you never set it, then it will be
#' `"channels_last"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit layer_dense return
#' @export
#' @family image preprocessing layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/image_preprocessing/center_crop#centercrop-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/CenterCrop>
#' @tether keras.layers.CenterCrop
layer_center_crop <-
function (object, height, width, data_format = NULL, ...)
{
    args <- capture_args(list(height = as_integer, width = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$CenterCrop, object, args)
}


#' A preprocessing layer which buckets continuous features by ranges.
#'
#' @description
#' This layer will place each element of its input data into one of several
#' contiguous ranges and output an integer index indicating which range each
#' element was placed in.
#'
#' **Note:** This layer is safe to use inside a `tf.data` pipeline
#' (independently of which backend you're using).
#'
#' # Input Shape
#' Any array of dimension 2 or higher.
#'
#' # Output Shape
#' Same as input shape.
#'
#' # Examples
#' Discretize float values based on provided buckets.
#' ```{r}
#' input <- op_array(rbind(c(-1.5, 1, 3.4, 0.5),
#'                        c(0, 3, 1.3, 0),
#'                        c(-.5, 0, .5, 1),
#'                        c(1.5, 2, 2.5, 3)))
#' output <- input |> layer_discretization(bin_boundaries = c(0, 1, 2))
#' output
#' ```
#'
#' Discretize float values based on a number of buckets to compute.
#' ```{r}
#' layer <- layer_discretization(num_bins = 4, epsilon = 0.01)
#' layer |> adapt(input)
#' layer(input)
#' ```
#'
#' @param bin_boundaries
#' A list of bin boundaries.
#' The leftmost and rightmost bins
#' will always extend to `-Inf` and `Inf`,
#' so `bin_boundaries = c(0, 1, 2)`
#' generates bins `(-Inf, 0)`, `[0, 1)`, `[1, 2)`,
#' and `[2, +Inf)`.
#' If this option is set, `adapt()` should not be called.
#'
#' @param num_bins
#' The integer number of bins to compute.
#' If this option is set,
#' `adapt()` should be called to learn the bin boundaries.
#'
#' @param epsilon
#' Error tolerance, typically a small fraction
#' close to zero (e.g. 0.01). Higher values of epsilon increase
#' the quantile approximation, and hence result in more
#' unequal buckets, but could improve performance
#' and resource consumption.
#'
#' @param output_mode
#' Specification for the output of the layer.
#' Values can be `"int"`, `"one_hot"`, `"multi_hot"`, or
#' `"count"` configuring the layer as follows:
#' - `"int"`: Return the discretized bin indices directly.
#' - `"one_hot"`: Encodes each individual element in the
#'     input into an array the same size as `num_bins`,
#'     containing a 1 at the input's bin
#'     index. If the last dimension is size 1, will encode on that
#'     dimension.  If the last dimension is not size 1,
#'     will append a new dimension for the encoded output.
#' - `"multi_hot"`: Encodes each sample in the input into a
#'     single array the same size as `num_bins`,
#'     containing a 1 for each bin index
#'     index present in the sample.
#'     Treats the last dimension as the sample
#'     dimension, if input shape is `(..., sample_length)`,
#'     output shape will be `(..., num_tokens)`.
#' - `"count"`: As `"multi_hot"`, but the int array contains
#'     a count of the number of times the bin index appeared
#'     in the sample.
#' Defaults to `"int"`.
#'
#' @param sparse
#' Boolean. Only applicable to `"one_hot"`, `"multi_hot"`,
#' and `"count"` output modes. Only supported with TensorFlow
#' backend. If `TRUE`, returns a `SparseTensor` instead of
#' a dense `Tensor`. Defaults to `FALSE`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param name
#' String, name for the object
#'
#' @param dtype
#' datatype (e.g., `"float32"`).
#'
#' @inherit layer_dense return
#' @export
#' @family numerical features preprocessing layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/numerical/discretization#discretization-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Discretization>
#'
#' @tether keras.layers.Discretization
layer_discretization <-
function (object, bin_boundaries = NULL, num_bins = NULL, epsilon = 0.01,
    output_mode = "int", sparse = FALSE, dtype = NULL, name = NULL)
{
    args <- capture_args(list(num_bins = as_integer, output_mode = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$Discretization, object, args)
}


#' A preprocessing layer which crosses features using the "hashing trick".
#'
#' @description
#' This layer performs crosses of categorical features using the "hashing
#' trick". Conceptually, the transformation can be thought of as:
#' `hash(concatenate(features)) %% num_bins`.
#'
#' This layer currently only performs crosses of scalar inputs and batches of
#' scalar inputs. Valid input shapes are `(batch_size, 1)`, `(batch_size)` and
#' `()`.
#'
#' **Note:** This layer wraps `tf.keras.layers.HashedCrossing`. It cannot
#' be used as part of the compiled computation graph of a model with
#' any backend other than TensorFlow.
#' It can however be used with any backend when running eagerly.
#' It can also always be used as part of an input preprocessing pipeline
#' with any backend (outside the model itself), which is how we recommend
#' to use this layer.
#'
#' **Note:** This layer is safe to use inside a `tfdatasets` pipeline
#' (independently of which backend you're using).
#'
#' # Examples
#'
#' ```{r}
#' feat1 <- c('A', 'B', 'A', 'B', 'A') |> as.array()
#' feat2 <- c(101, 101, 101, 102, 102) |> as.integer() |> as.array()
#' ```
#'
#' **Crossing two scalar features.**
#'
#' ```{r}
#' layer <- layer_hashed_crossing(num_bins = 5)
#' layer(list(feat1, feat2))
#' ```
#'
#' **Crossing and one-hotting two scalar features.**
#'
#' ```{r}
#' layer <- layer_hashed_crossing(num_bins = 5, output_mode = 'one_hot')
#' layer(list(feat1, feat2))
#' ```
#'
#' @param num_bins
#' Number of hash bins.
#'
#' @param output_mode
#' Specification for the output of the layer. Values can be
#' `"int"`, or `"one_hot"` configuring the layer as follows:
#' - `"int"`: Return the integer bin indices directly.
#' - `"one_hot"`: Encodes each individual element in the input into an
#'     array the same size as `num_bins`, containing a 1 at the input's
#'     bin index. Defaults to `"int"`.
#'
#' @param sparse
#' Boolean. Only applicable to `"one_hot"` mode and only valid
#' when using the TensorFlow backend. If `TRUE`, returns
#' a `SparseTensor` instead of a dense `Tensor`. Defaults to `FALSE`.
#'
#' @param ...
#' Keyword arguments to construct a layer.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param name
#' String, name for the object
#'
#' @param dtype
#' datatype (e.g., `"float32"`).
#'
#' @inherit layer_dense return
#' @export
#' @family categorical features preprocessing layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/categorical/hashed_crossing#hashedcrossing-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/HashedCrossing>
#'
#' @tether keras.layers.HashedCrossing
layer_hashed_crossing <-
function (object, num_bins, output_mode = "int", sparse = FALSE,
    name = NULL, dtype = NULL, ...)
{
    args <- capture_args(list(output_mode = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape,
        num_bins = as_integer), ignore = "object")
    create_layer(keras$layers$HashedCrossing, object, args)
}


#' A preprocessing layer which hashes and bins categorical features.
#'
#' @description
#' This layer transforms categorical inputs to hashed output. It element-wise
#' converts a ints or strings to ints in a fixed range. The stable hash
#' function uses `tensorflow::ops::Fingerprint` to produce the same output
#' consistently across all platforms.
#'
#' This layer uses [FarmHash64](https://github.com/google/farmhash) by default,
#' which provides a consistent hashed output across different platforms and is
#' stable across invocations, regardless of device and context, by mixing the
#' input bits thoroughly.
#'
#' If you want to obfuscate the hashed output, you can also pass a random
#' `salt` argument in the constructor. In that case, the layer will use the
#' [SipHash64](https://github.com/google/highwayhash) hash function, with
#' the `salt` value serving as additional input to the hash function.
#'
#' **Note:** This layer internally uses TensorFlow. It cannot
#' be used as part of the compiled computation graph of a model with
#' any backend other than TensorFlow.
#' It can however be used with any backend when running eagerly.
#' It can also always be used as part of an input preprocessing pipeline
#' with any backend (outside the model itself), which is how we recommend
#' to use this layer.
#'
#' **Note:** This layer is safe to use inside a `tf.data` pipeline
#' (independently of which backend you're using).
#'
#' **Example (FarmHash64)**
#'
#' ```{r}
#' layer <- layer_hashing(num_bins = 3)
#' inp <- c('A', 'B', 'C', 'D', 'E') |> array(dim = c(5, 1))
#' layer(inp)
#' ```
#'
#' **Example (FarmHash64) with a mask value**
#'
#' ```{r}
#' layer <- layer_hashing(num_bins=3, mask_value='')
#' inp <- c('A', 'B', '', 'C', 'D') |> array(dim = c(5, 1))
#' layer(inp)
#' ```
#'
#' **Example (SipHash64)**
#'
#' ```{r}
#' layer <- layer_hashing(num_bins=3, salt=c(133, 137))
#' inp <- c('A', 'B', 'C', 'D', 'E') |> array(dim = c(5, 1))
#' layer(inp)
#' ```
#'
#' **Example (Siphash64 with a single integer, same as `salt=[133, 133]`)**
#'
#' ```{r}
#' layer <- layer_hashing(num_bins=3, salt=133)
#' inp <- c('A', 'B', 'C', 'D', 'E') |> array(dim = c(5, 1))
#' layer(inp)
#' ```
#'
#' # Input Shape
#' A single string, a list of strings, or an `int32` or `int64` tensor
#' of shape `(batch_size, ...,)`.
#'
#' # Output Shape
#' An `int32` tensor of shape `(batch_size, ...)`.
#'
#' # Reference
#' - [SipHash with salt](https://en.wikipedia.org/wiki/SipHash)
#'
#' @param num_bins
#' Number of hash bins. Note that this includes the `mask_value`
#' bin, so the effective number of bins is `(num_bins - 1)`
#' if `mask_value` is set.
#'
#' @param mask_value
#' A value that represents masked inputs, which are mapped to
#' index 0. `NULL` means no mask term will be added and the
#' hashing will start at index 0. Defaults to `NULL`.
#'
#' @param salt
#' A single unsigned integer or `NULL`.
#' If passed, the hash function used will be SipHash64,
#' with these values used as an additional input
#' (known as a "salt" in cryptography).
#' These should be non-zero. If `NULL`, uses the FarmHash64 hash
#' function. It also supports list of 2 unsigned
#' integer numbers, see reference paper for details.
#' Defaults to `NULL`.
#'
#' @param output_mode
#' Specification for the output of the layer. Values can be
#' `"int"`, `"one_hot"`, `"multi_hot"`, or
#' `"count"` configuring the layer as follows:
#' - `"int"`: Return the integer bin indices directly.
#' - `"one_hot"`: Encodes each individual element in the input into an
#'     array the same size as `num_bins`, containing a 1
#'     at the input's bin index. If the last dimension is size 1,
#'     will encode on that dimension.
#'     If the last dimension is not size 1, will append a new
#'     dimension for the encoded output.
#' - `"multi_hot"`: Encodes each sample in the input into a
#'     single array the same size as `num_bins`,
#'     containing a 1 for each bin index
#'     index present in the sample. Treats the last dimension
#'     as the sample dimension, if input shape is
#'     `(..., sample_length)`, output shape will be
#'     `(..., num_tokens)`.
#' - `"count"`: As `"multi_hot"`, but the int array contains a count of
#'     the number of times the bin index appeared in the sample.
#' Defaults to `"int"`.
#'
#' @param sparse
#' Boolean. Only applicable to `"one_hot"`, `"multi_hot"`,
#' and `"count"` output modes. Only supported with TensorFlow
#' backend. If `TRUE`, returns a `SparseTensor` instead of
#' a dense `Tensor`. Defaults to `FALSE`.
#'
#' @param ...
#' Keyword arguments to construct a layer.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @inherit layer_dense return
#' @export
#' @family categorical features preprocessing layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/categorical/hashing#hashing-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Hashing>
#'
#' @tether keras.layers.Hashing
layer_hashing <-
function (object, num_bins, mask_value = NULL, salt = NULL, output_mode = "int",
    sparse = FALSE, ...)
{
    args <- capture_args(list(salt = as_integer, output_mode = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape, num_bins = as_integer),
        ignore = "object")
    create_layer(keras$layers$Hashing, object, args)
}


#' A preprocessing layer that maps integers to (possibly encoded) indices.
#'
#' @description
#' This layer maps a set of arbitrary integer input tokens into indexed integer
#' output via a table-based vocabulary lookup. The layer's output indices will
#' be contiguously arranged up to the maximum vocab size, even if the input
#' tokens are non-continguous or unbounded. The layer supports multiple options
#' for encoding the output via `output_mode`, and has optional support for
#' out-of-vocabulary (OOV) tokens and masking.
#'
#' The vocabulary for the layer must be either supplied on construction or
#' learned via `adapt()`. During `adapt()`, the layer will analyze a data set,
#' determine the frequency of individual integer tokens, and create a
#' vocabulary from them. If the vocabulary is capped in size, the most frequent
#' tokens will be used to create the vocabulary and all others will be treated
#' as OOV.
#'
#' There are two possible output modes for the layer.  When `output_mode` is
#' `"int"`, input integers are converted to their index in the vocabulary (an
#' integer).  When `output_mode` is `"multi_hot"`, `"count"`, or `"tf_idf"`,
#' input integers are encoded into an array where each dimension corresponds to
#' an element in the vocabulary.
#'
#' The vocabulary can optionally contain a mask token as well as an OOV token
#' (which can optionally occupy multiple indices in the vocabulary, as set
#' by `num_oov_indices`).
#' The position of these tokens in the vocabulary is fixed. When `output_mode`
#' is `"int"`, the vocabulary will begin with the mask token at index 0,
#' followed by OOV indices, followed by the rest of the vocabulary. When
#' `output_mode` is `"multi_hot"`, `"count"`, or `"tf_idf"` the vocabulary will
#' begin with OOV indices and instances of the mask token will be dropped.
#'
#' **Note:** This layer uses TensorFlow internally. It cannot
#' be used as part of the compiled computation graph of a model with
#' any backend other than TensorFlow.
#' It can however be used with any backend when running eagerly.
#' It can also always be used as part of an input preprocessing pipeline
#' with any backend (outside the model itself), which is how we recommend
#' to use this layer.
#'
#' **Note:** This layer is safe to use inside a `tf.data` pipeline
#' (independently of which backend you're using).
#'
#' # Examples
#' **Creating a lookup layer with a known vocabulary**
#'
#' This example creates a lookup layer with a pre-existing vocabulary.
#'
#' ```{r}
#' vocab <- c(12, 36, 1138, 42) |> as.integer()
#' data <- op_array(rbind(c(12, 1138, 42),
#'                       c(42, 1000, 36)))  # Note OOV tokens
#' out <- data |> layer_integer_lookup(vocabulary = vocab)
#' out
#' ```
#'
#' **Creating a lookup layer with an adapted vocabulary**
#'
#' This example creates a lookup layer and generates the vocabulary by
#' analyzing the dataset.
#'
#' ```{r}
#' data <- op_array(rbind(c(12, 1138, 42),
#'                       c(42, 1000, 36)))  # Note OOV tokens
#' layer <- layer_integer_lookup()
#' layer |> adapt(data)
#' layer |> get_vocabulary() |> str()
#' ```
#'
#' Note that the OOV token -1 have been added to the vocabulary. The remaining
#' tokens are sorted by frequency (42, which has 2 occurrences, is first) then
#' by inverse sort order.
#'
#' ```{r}
#' layer(data)
#' ```
#'
#' **Lookups with multiple OOV indices**
#'
#' This example demonstrates how to use a lookup layer with multiple OOV
#' indices.  When a layer is created with more than one OOV index, any OOV
#' tokens are hashed into the number of OOV buckets, distributing OOV tokens in
#' a deterministic fashion across the set.
#'
#' ```{r}
#' vocab <- c(12, 36, 1138, 42) |> as.integer()
#' data <- op_array(rbind(c(12, 1138, 42),
#'                       c(37, 1000, 36)))  # Note OOV tokens
#' out <- data |>
#'   layer_integer_lookup(vocabulary = vocab,
#'                        num_oov_indices = 2)
#' out
#' ```
#'
#' Note that the output for OOV token 37 is 1, while the output for OOV token
#' 1000 is 0. The in-vocab terms have their output index increased by 1 from
#' earlier examples (12 maps to 2, etc) in order to make space for the extra
#' OOV token.
#'
#' **One-hot output**
#'
#' Configure the layer with `output_mode='one_hot'`. Note that the first
#' `num_oov_indices` dimensions in the ont_hot encoding represent OOV values.
#'
#' ```{r}
#' vocab <- c(12, 36, 1138, 42) |> as.integer()
#' data <- op_array(c(12, 36, 1138, 42, 7), 'int32')  # Note OOV tokens
#' layer <- layer_integer_lookup(vocabulary = vocab,
#'                               output_mode = 'one_hot')
#' layer(data)
#' ```
#'
#' **Multi-hot output**
#'
#' Configure the layer with `output_mode = 'multi_hot'`. Note that the first
#' `num_oov_indices` dimensions in the multi_hot encoding represent OOV tokens
#'
#' ```{r}
#' vocab <- c(12, 36, 1138, 42) |> as.integer()
#' data <- op_array(rbind(c(12, 1138, 42, 42),
#'                       c(42,    7, 36,  7)), "int64")  # Note OOV tokens
#' layer <- layer_integer_lookup(vocabulary = vocab,
#'                               output_mode = 'multi_hot')
#' layer(data)
#' ```
#'
#' **Token count output**
#'
#' Configure the layer with `output_mode='count'`. As with multi_hot output,
#' the first `num_oov_indices` dimensions in the output represent OOV tokens.
#'
#' ```{r}
#' vocab <- c(12, 36, 1138, 42) |> as.integer()
#' data <- rbind(c(12, 1138, 42, 42),
#'               c(42,    7, 36,  7)) |> op_array("int64")
#' layer <- layer_integer_lookup(vocabulary = vocab,
#'                               output_mode = 'count')
#' layer(data)
#' ```
#'
#' **TF-IDF output**
#'
#' Configure the layer with `output_mode='tf_idf'`. As with multi_hot output,
#' the first `num_oov_indices` dimensions in the output represent OOV tokens.
#'
#' Each token bin will output `token_count * idf_weight`, where the idf weights
#' are the inverse document frequency weights per token. These should be
#' provided along with the vocabulary. Note that the `idf_weight` for OOV
#' tokens will default to the average of all idf weights passed in.
#'
#' ```{r}
#' vocab <- c(12, 36, 1138, 42) |> as.integer()
#' idf_weights <- c(0.25, 0.75, 0.6, 0.4)
#' data <- rbind(c(12, 1138, 42, 42),
#'               c(42,    7, 36,  7)) |> op_array("int64")
#' layer <- layer_integer_lookup(output_mode = 'tf_idf',
#'                               vocabulary = vocab,
#'                               idf_weights = idf_weights)
#' layer(data)
#' ```
#'
#' To specify the idf weights for oov tokens, you will need to pass the entire
#' vocabulary including the leading oov token.
#'
#' ```{r}
#' vocab <- c(-1, 12, 36, 1138, 42) |> as.integer()
#' idf_weights <- c(0.9, 0.25, 0.75, 0.6, 0.4)
#' data <- rbind(c(12, 1138, 42, 42),
#'               c(42,    7, 36,  7)) |> op_array("int64")
#' layer <- layer_integer_lookup(output_mode = 'tf_idf',
#'                               vocabulary = vocab,
#'                               idf_weights = idf_weights)
#' layer(data)
#' ```
#'
#' When adapting the layer in `"tf_idf"` mode, each input sample will
#' be considered a document, and IDF weight per token will be
#' calculated as:
#' `log(1 + num_documents / (1 + token_document_count))`.
#'
#' **Inverse lookup**
#'
#' This example demonstrates how to map indices to tokens using this layer.
#' (You can also use `adapt()` with `inverse = TRUE`, but for simplicity we'll
#' pass the vocab in this example.)
#'
#' ```{r}
#' vocab <- c(12, 36, 1138, 42) |> as.integer()
#' data <- op_array(c(1, 3, 4,
#'                   4, 0, 2)) |> op_reshape(c(2,-1)) |> op_cast("int32")
#' layer <- layer_integer_lookup(vocabulary = vocab, invert = TRUE)
#' layer(data)
#' ```
#'
#' Note that the first index correspond to the oov token by default.
#'
#' **Forward and inverse lookup pairs**
#'
#' This example demonstrates how to use the vocabulary of a standard lookup
#' layer to create an inverse lookup layer.
#'
#' ```{r}
#' vocab <- c(12, 36, 1138, 42) |> as.integer()
#' data <- op_array(rbind(c(12, 1138, 42), c(42, 1000, 36)), "int32")
#' layer <- layer_integer_lookup(vocabulary = vocab)
#' i_layer <- layer_integer_lookup(vocabulary = get_vocabulary(layer),
#'                                 invert = TRUE)
#' int_data <- layer(data)
#' i_layer(int_data)
#' ```
#'
#' In this example, the input token 1000 resulted in an output of -1, since
#' 1000 was not in the vocabulary - it got represented as an OOV, and all OOV
#' tokens are returned as -1 in the inverse layer. Also, note that for the
#' inverse to work, you must have already set the forward layer vocabulary
#' either directly or via `adapt()` before calling `get_vocabulary()`.
#'
#' @param max_tokens
#' Maximum size of the vocabulary for this layer. This should
#' only be specified when adapting the vocabulary or when setting
#' `pad_to_max_tokens=TRUE`. If NULL, there is no cap on the size of
#' the vocabulary. Note that this size includes the OOV
#' and mask tokens. Defaults to `NULL`.
#'
#' @param num_oov_indices
#' The number of out-of-vocabulary tokens to use.
#' If this value is more than 1, OOV inputs are modulated to
#' determine their OOV value.
#' If this value is 0, OOV inputs will cause an error when calling
#' the layer. Defaults to `1`.
#'
#' @param mask_token
#' An integer token that represents masked inputs. When
#' `output_mode` is `"int"`, the token is included in vocabulary
#' and mapped to index 0. In other output modes,
#' the token will not appear in the vocabulary and instances
#' of the mask token in the input will be dropped.
#' If set to NULL, no mask term will be added. Defaults to `NULL`.
#'
#' @param oov_token
#' Only used when `invert` is `TRUE`. The token to return
#' for OOV indices. Defaults to `-1`.
#'
#' @param vocabulary
#' Optional. Either an array of integers or a string path to a
#' text file. If passing an array, can pass a list, list,
#' 1D NumPy array, or 1D tensor containing the integer vocbulary terms.
#' If passing a file path, the file should contain one line per term
#' in the vocabulary. If this argument is set,
#' there is no need to `adapt()` the layer.
#'
#' @param vocabulary_dtype
#' The dtype of the vocabulary terms, for example
#' `"int64"` or `"int32"`. Defaults to `"int64"`.
#'
#' @param idf_weights
#' Only valid when `output_mode` is `"tf_idf"`.
#' A list, list, 1D NumPy array, or 1D tensor or the same length
#' as the vocabulary, containing the floating point inverse document
#' frequency weights, which will be multiplied by per sample term
#' counts for the final TF-IDF weight.
#' If the `vocabulary` argument is set, and `output_mode` is
#' `"tf_idf"`, this argument must be supplied.
#'
#' @param invert
#' Only valid when `output_mode` is `"int"`.
#' If `TRUE`, this layer will map indices to vocabulary items
#' instead of mapping vocabulary items to indices.
#' Defaults to `FALSE`.
#'
#' @param output_mode
#' Specification for the output of the layer. Values can be
#' `"int"`, `"one_hot"`, `"multi_hot"`, `"count"`, or `"tf_idf"`
#' configuring the layer as follows:
#' - `"int"`: Return the vocabulary indices of the input tokens.
#' - `"one_hot"`: Encodes each individual element in the input into an
#'     array the same size as the vocabulary,
#'     containing a 1 at the element index. If the last dimension
#'     is size 1, will encode on that dimension.
#'     If the last dimension is not size 1, will append a new
#'     dimension for the encoded output.
#' - `"multi_hot"`: Encodes each sample in the input into a single
#'     array the same size as the vocabulary,
#'     containing a 1 for each vocabulary term present in the sample.
#'     Treats the last dimension as the sample dimension,
#'     if input shape is `(..., sample_length)`,
#'     output shape will be `(..., num_tokens)`.
#' - `"count"`: As `"multi_hot"`, but the int array contains
#'     a count of the number of times the token at that index
#'     appeared in the sample.
#' - `"tf_idf"`: As `"multi_hot"`, but the TF-IDF algorithm is
#'     applied to find the value in each token slot.
#' For `"int"` output, any shape of input and output is supported.
#' For all other output modes, currently only output up to rank 2
#' is supported. Defaults to `"int"`.
#'
#' @param pad_to_max_tokens
#' Only applicable when `output_mode` is `"multi_hot"`,
#' `"count"`, or `"tf_idf"`. If `TRUE`, the output will have
#' its feature axis padded to `max_tokens` even if the number
#' of unique tokens in the vocabulary is less than `max_tokens`,
#' resulting in a tensor of shape `(batch_size, max_tokens)`
#' regardless of vocabulary size. Defaults to `FALSE`.
#'
#' @param sparse
#' Boolean. Only applicable to `"multi_hot"`, `"count"`, and
#' `"tf_idf"` output modes. Only supported with TensorFlow
#' backend. If `TRUE`, returns a `SparseTensor`
#' instead of a dense `Tensor`. Defaults to `FALSE`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param name
#' String, name for the object
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit layer_dense return
#' @export
#' @family categorical features preprocessing layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/categorical/integer_lookup#integerlookup-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/IntegerLookup>
#'
#' @tether keras.layers.IntegerLookup
layer_integer_lookup <-
function (object, max_tokens = NULL, num_oov_indices = 1L, mask_token = NULL,
    oov_token = -1L, vocabulary = NULL, vocabulary_dtype = "int64",
    idf_weights = NULL, invert = FALSE, output_mode = "int",
    sparse = FALSE, pad_to_max_tokens = FALSE, name = NULL, ...)
{
    args <- capture_args(list(num_oov_indices = as_integer,
        mask_token = as_integer, oov_token = as_integer, vocabulary = as_integer,
        invert = as_integer, output_mode = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$IntegerLookup, object, args)
}


#' A preprocessing layer that normalizes continuous features.
#'
#' @description
#' This layer will shift and scale inputs into a distribution centered around
#' 0 with standard deviation 1. It accomplishes this by precomputing the mean
#' and variance of the data, and calling `(input - mean) / sqrt(var)` at
#' runtime.
#'
#' The mean and variance values for the layer must be either supplied on
#' construction or learned via `adapt()`. `adapt()` will compute the mean and
#' variance of the data and store them as the layer's weights. `adapt()` should
#' be called before `fit()`, `evaluate()`, or `predict()`.
#'
#' # Examples
#' Calculate a global mean and variance by analyzing the dataset in `adapt()`.
#'
#' ```{r}
#' adapt_data <- op_array(c(1., 2., 3., 4., 5.), dtype='float32')
#' input_data <- op_array(c(1., 2., 3.), dtype='float32')
#' layer <- layer_normalization(axis = NULL)
#' layer %>% adapt(adapt_data)
#' layer(input_data)
#' ```
#'
#' Calculate a mean and variance for each index on the last axis.
#'
#' ```{r}
#' adapt_data <- op_array(rbind(c(0., 7., 4.),
#'                        c(2., 9., 6.),
#'                        c(0., 7., 4.),
#'                        c(2., 9., 6.)), dtype='float32')
#' input_data <- op_array(matrix(c(0., 7., 4.), nrow = 1), dtype='float32')
#' layer <- layer_normalization(axis=-1)
#' layer %>% adapt(adapt_data)
#' layer(input_data)
#' ```
#'
#' Pass the mean and variance directly.
#'
#' ```{r}
#' input_data <- op_array(rbind(1, 2, 3), dtype='float32')
#' layer <- layer_normalization(mean=3., variance=2.)
#' layer(input_data)
#' ```
#'
#' Use the layer to de-normalize inputs (after adapting the layer).
#'
#' ```{r}
#' adapt_data <- op_array(rbind(c(0., 7., 4.),
#'                        c(2., 9., 6.),
#'                        c(0., 7., 4.),
#'                        c(2., 9., 6.)), dtype='float32')
#' input_data <- op_array(c(1., 2., 3.), dtype='float32')
#' layer <- layer_normalization(axis=-1, invert=TRUE)
#' layer %>% adapt(adapt_data)
#' layer(input_data)
#' ```
#'
#' @param axis
#' Integer, list of integers, or NULL. The axis or axes that should
#' have a separate mean and variance for each index in the shape.
#' For example, if shape is `(NULL, 5)` and `axis=1`, the layer will
#' track 5 separate mean and variance values for the last axis.
#' If `axis` is set to `NULL`, the layer will normalize
#' all elements in the input by a scalar mean and variance.
#' When `-1`, the last axis of the input is assumed to be a
#' feature dimension and is normalized per index.
#' Note that in the specific case of batched scalar inputs where
#' the only axis is the batch axis, the default will normalize
#' each index in the batch separately.
#' In this case, consider passing `axis=NULL`. Defaults to `-1`.
#'
#' @param mean
#' The mean value(s) to use during normalization. The passed value(s)
#' will be broadcast to the shape of the kept axes above;
#' if the value(s) cannot be broadcast, an error will be raised when
#' this layer's `build()` method is called.
#'
#' @param variance
#' The variance value(s) to use during normalization. The passed
#' value(s) will be broadcast to the shape of the kept axes above;
#' if the value(s) cannot be broadcast, an error will be raised when
#' this layer's `build()` method is called.
#'
#' @param invert
#' If `TRUE`, this layer will apply the inverse transformation
#' to its inputs: it would turn a normalized input back into its
#' original form.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit layer_dense return
#' @export
#' @family numerical features preprocessing layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/numerical/normalization#normalization-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization>
#'
#' @tether keras.layers.Normalization
layer_normalization <-
function (object, axis = -1L, mean = NULL, variance = NULL, invert = FALSE,
    ...)
{
    args <- capture_args(list(axis = as_axis, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$Normalization, object, args)
}


#' A preprocessing layer which randomly adjusts brightness during training.
#'
#' @description
#' This layer will randomly increase/reduce the brightness for the input RGB
#' images. At inference time, the output will be identical to the input.
#' Call the layer with `training=TRUE` to adjust the brightness of the input.
#'
#' **Note:** This layer is safe to use inside a `tf.data` pipeline
#' (independently of which backend you're using).
#'
#' # Inputs
#' 3D (HWC) or 4D (NHWC) tensor, with float or int dtype. Input pixel
#' values can be of any range (e.g. `[0., 1.)` or `[0, 255]`)
#'
#' # Output
#' 3D (HWC) or 4D (NHWC) tensor with brightness adjusted based on the
#'     `factor`. By default, the layer will output floats.
#'     The output value will be clipped to the range `[0, 255]`,
#'     the valid range of RGB colors, and
#'     rescaled based on the `value_range` if needed.
#'
#' # Example
#'
#' ```{r}
#' random_bright <- layer_random_brightness(factor=0.2, seed = 1)
#'
#' # An image with shape [2, 2, 3]
#' image <- array(1:12, dim=c(2, 2, 3))
#'
#' # Assume we randomly select the factor to be 0.1, then it will apply
#' # 0.1 * 255 to all the channel
#' output <- random_bright(image, training=TRUE)
#' output
#' ```
#'
#' @param factor
#' Float or a list of 2 floats between -1.0 and 1.0. The
#' factor is used to determine the lower bound and upper bound of the
#' brightness adjustment. A float value will be chosen randomly between
#' the limits. When -1.0 is chosen, the output image will be black, and
#' when 1.0 is chosen, the image will be fully white.
#' When only one float is provided, eg, 0.2,
#' then -0.2 will be used for lower bound and 0.2
#' will be used for upper bound.
#'
#' @param value_range
#' Optional list of 2 floats
#' for the lower and upper limit
#' of the values of the input data.
#' To make no change, use `c(0.0, 1.0)`, e.g., if the image input
#' has been scaled before this layer. Defaults to `c(0.0, 255.0)`.
#' The brightness adjustment will be scaled to this range, and the
#' output values will be clipped to this range.
#'
#' @param seed
#' optional integer, for fixed RNG behavior.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit layer_dense return
#' @export
#' @family image augmentation layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_brightness#randombrightness-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomBrightness>
#'
#' @tether keras.layers.RandomBrightness
layer_random_brightness <-
function (object, factor, value_range = list(0L, 255L), seed = NULL,
    ...)
{
    args <- capture_args(list(seed = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$RandomBrightness, object, args)
}


#' A preprocessing layer which randomly adjusts contrast during training.
#'
#' @description
#' This layer will randomly adjust the contrast of an image or images
#' by a random factor. Contrast is adjusted independently
#' for each channel of each image during training.
#'
#' For each channel, this layer computes the mean of the image pixels in the
#' channel and then adjusts each component `x` of each pixel to
#' `(x - mean) * contrast_factor + mean`.
#'
#' Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
#' in integer or floating point dtype.
#' By default, the layer will output floats.
#'
#' **Note:** This layer is safe to use inside a `tf.data` pipeline
#' (independently of which backend you're using).
#'
#' # Input Shape
#' 3D (unbatched) or 4D (batched) tensor with shape:
#' `(..., height, width, channels)`, in `"channels_last"` format.
#'
#' # Output Shape
#' 3D (unbatched) or 4D (batched) tensor with shape:
#' `(..., height, width, channels)`, in `"channels_last"` format.
#'
#' @param factor
#' a positive float represented as fraction of value, or a tuple of
#' size 2 representing lower and upper bound.
#' When represented as a single float, lower = upper.
#' The contrast factor will be randomly picked between
#' `[1.0 - lower, 1.0 + upper]`. For any pixel x in the channel,
#' the output will be `(x - mean) * factor + mean`
#' where `mean` is the mean value of the channel.
#'
#' @param seed
#' Integer. Used to create a random seed.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit layer_dense return
#' @export
#' @family image augmentation layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_contrast#randomcontrast-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomContrast>
#' @tether keras.layers.RandomContrast
layer_random_contrast <-
function (object, factor, seed = NULL, ...)
{
    args <- capture_args(list(seed = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$RandomContrast, object, args)
}


#' A preprocessing layer which randomly crops images during training.
#'
#' @description
#' During training, this layer will randomly choose a location to crop images
#' down to a target size. The layer will crop all the images in the same batch
#' to the same cropping location.
#'
#' At inference time, and during training if an input image is smaller than the
#' target size, the input will be resized and cropped so as to return the
#' largest possible window in the image that matches the target aspect ratio.
#' If you need to apply random cropping at inference time, set `training` to
#' TRUE when calling the layer.
#'
#' Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
#' of integer or floating point dtype. By default, the layer will output
#' floats.
#'
#' **Note:** This layer is safe to use inside a `tf.data` pipeline
#' (independently of which backend you're using).
#'
#' # Input Shape
#' 3D (unbatched) or 4D (batched) tensor with shape:
#' `(..., height, width, channels)`, in `"channels_last"` format.
#'
#' # Output Shape
#' 3D (unbatched) or 4D (batched) tensor with shape:
#' `(..., target_height, target_width, channels)`.
#'
#' @param height
#' Integer, the height of the output shape.
#'
#' @param width
#' Integer, the width of the output shape.
#'
#' @param seed
#' Integer. Used to create a random seed.
#'
#' @param ...
#' Base layer keyword arguments, such as
#' `name` and `dtype`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param name
#' String, name for the object
#'
#' @param data_format
#' see description
#'
#' @inherit layer_dense return
#' @export
#' @family image augmentation layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_crop#randomcrop-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomCrop>
#' @tether keras.layers.RandomCrop
layer_random_crop <-
function (object, height, width, seed = NULL, data_format = NULL,
    name = NULL, ...)
{
    args <- capture_args(list(height = as_integer, width = as_integer,
        seed = as_integer, input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$RandomCrop, object, args)
}


#' A preprocessing layer which randomly flips images during training.
#'
#' @description
#' This layer will flip the images horizontally and or vertically based on the
#' `mode` attribute. During inference time, the output will be identical to
#' input. Call the layer with `training=TRUE` to flip the input.
#' Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
#' of integer or floating point dtype.
#' By default, the layer will output floats.
#'
#' **Note:** This layer is safe to use inside a `tf.data` pipeline
#' (independently of which backend you're using).
#'
#' # Input Shape
#' 3D (unbatched) or 4D (batched) tensor with shape:
#' `(..., height, width, channels)`, in `"channels_last"` format.
#'
#' # Output Shape
#' 3D (unbatched) or 4D (batched) tensor with shape:
#' `(..., height, width, channels)`, in `"channels_last"` format.
#'
#' @param mode
#' String indicating which flip mode to use. Can be `"horizontal"`,
#' `"vertical"`, or `"horizontal_and_vertical"`. `"horizontal"` is a
#' left-right flip and `"vertical"` is a top-bottom flip. Defaults to
#' `"horizontal_and_vertical"`
#'
#' @param seed
#' Integer. Used to create a random seed.
#'
#' @param ...
#' Base layer keyword arguments, such as
#' `name` and `dtype`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @inherit layer_dense return
#' @export
#' @family image augmentation layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_flip#randomflip-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomFlip>
#' @tether keras.layers.RandomFlip
layer_random_flip <-
function (object, mode = "horizontal_and_vertical", seed = NULL,
    ...)
{
    args <- capture_args(list(seed = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$RandomFlip, object, args)
}


#' A preprocessing layer which randomly rotates images during training.
#'
#' @description
#' This layer will apply random rotations to each image, filling empty space
#' according to `fill_mode`.
#'
#' By default, random rotations are only applied during training.
#' At inference time, the layer does nothing. If you need to apply random
#' rotations at inference time, pass `training = TRUE` when calling the layer.
#'
#' Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
#' of integer or floating point dtype.
#' By default, the layer will output floats.
#'
#' **Note:** This layer is safe to use inside a `tf.data` pipeline
#' (independently of which backend you're using).
#'
#' # Input Shape
#' 3D (unbatched) or 4D (batched) tensor with shape:
#' `(..., height, width, channels)`, in `"channels_last"` format
#'
#' # Output Shape
#' 3D (unbatched) or 4D (batched) tensor with shape:
#' `(..., height, width, channels)`, in `"channels_last"` format
#'
#' @param factor
#' a float represented as fraction of 2 Pi, or a tuple of size 2
#' representing lower and upper bound for rotating clockwise and
#' counter-clockwise. A positive values means rotating
#' counter clock-wise,
#' while a negative value means clock-wise.
#' When represented as a single
#' float, this value is used for both the upper and lower bound.
#' For instance, `factor=(-0.2, 0.3)`
#' results in an output rotation by a random
#' amount in the range `[-20% * 2pi, 30% * 2pi]`.
#' `factor=0.2` results in an
#' output rotating by a random amount
#' in the range `[-20% * 2pi, 20% * 2pi]`.
#'
#' @param fill_mode
#' Points outside the boundaries of the input are filled
#' according to the given mode
#' (one of `{"constant", "reflect", "wrap", "nearest"}`).
#' - *reflect*: `(d c b a | a b c d | d c b a)`
#'     The input is extended by reflecting about
#'     the edge of the last pixel.
#' - *constant*: `(k k k k | a b c d | k k k k)`
#'     The input is extended by
#'     filling all values beyond the edge with
#'     the same constant value k = 0.
#' - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
#'     wrapping around to the opposite edge.
#' - *nearest*: `(a a a a | a b c d | d d d d)`
#'     The input is extended by the nearest pixel.
#'
#' @param interpolation
#' Interpolation mode. Supported values: `"nearest"`,
#' `"bilinear"`.
#'
#' @param seed
#' Integer. Used to create a random seed.
#'
#' @param fill_value
#' a float represents the value to be filled outside
#' the boundaries when `fill_mode="constant"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @param value_range
#' see description
#'
#' @param data_format
#' see description
#'
#' @inherit layer_dense return
#' @export
#' @family image augmentation layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_rotation#randomrotation-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomRotation>
#' @tether keras.layers.RandomRotation
layer_random_rotation <-
function (object, factor, fill_mode = "reflect", interpolation = "bilinear",
    seed = NULL, fill_value = 0, value_range = list(0L, 255L),
    data_format = NULL, ...)
{
    args <- capture_args(list(seed = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$RandomRotation, object, args)
}


#' A preprocessing layer which randomly translates images during training.
#'
#' @description
#' This layer will apply random translations to each image during training,
#' filling empty space according to `fill_mode`.
#'
#' Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
#' of integer or floating point dtype. By default, the layer will output
#' floats.
#'
#' # Input Shape
#' 3D (unbatched) or 4D (batched) tensor with shape:
#' `(..., height, width, channels)`, in `"channels_last"` format,
#' or `(..., channels, height, width)`, in `"channels_first"` format.
#'
#' # Output Shape
#' 3D (unbatched) or 4D (batched) tensor with shape:
#'     `(..., target_height, target_width, channels)`,
#'     or `(..., channels, target_height, target_width)`,
#'     in `"channels_first"` format.
#'
#' **Note:** This layer is safe to use inside a `tf.data` pipeline
#' (independently of which backend you're using).
#'
#' @param height_factor
#' a float represented as fraction of value, or a tuple of
#' size 2 representing lower and upper bound for shifting vertically. A
#' negative value means shifting image up, while a positive value means
#' shifting image down. When represented as a single positive float,
#' this value is used for both the upper and lower bound. For instance,
#' `height_factor=(-0.2, 0.3)` results in an output shifted by a random
#' amount in the range `[-20%, +30%]`. `height_factor=0.2` results in
#' an output height shifted by a random amount in the range
#' `[-20%, +20%]`.
#'
#' @param width_factor
#' a float represented as fraction of value, or a tuple of
#' size 2 representing lower and upper bound for shifting horizontally.
#' A negative value means shifting image left, while a positive value
#' means shifting image right. When represented as a single positive
#' float, this value is used for both the upper and lower bound. For
#' instance, `width_factor=(-0.2, 0.3)` results in an output shifted
#' left by 20%, and shifted right by 30%. `width_factor=0.2` results
#' in an output height shifted left or right by 20%.
#'
#' @param fill_mode
#' Points outside the boundaries of the input are filled
#' according to the given mode. Available methods are `"constant"`,
#' `"nearest"`, `"wrap"` and `"reflect"`. Defaults to `"constant"`.
#' - `"reflect"`: `(d c b a | a b c d | d c b a)`
#'     The input is extended by reflecting about the edge of the last
#'     pixel.
#' - `"constant"`: `(k k k k | a b c d | k k k k)`
#'     The input is extended by filling all values beyond
#'     the edge with the same constant value k specified by
#'     `fill_value`.
#' - `"wrap"`: `(a b c d | a b c d | a b c d)`
#'     The input is extended by wrapping around to the opposite edge.
#' - `"nearest"`: `(a a a a | a b c d | d d d d)`
#'     The input is extended by the nearest pixel.
#' Note that when using torch backend, `"reflect"` is redirected to
#' `"mirror"` `(c d c b | a b c d | c b a b)` because torch does not
#' support `"reflect"`.
#' Note that torch backend does not support `"wrap"`.
#'
#' @param interpolation
#' Interpolation mode. Supported values: `"nearest"`,
#' `"bilinear"`.
#'
#' @param seed
#' Integer. Used to create a random seed.
#'
#' @param fill_value
#' a float represents the value to be filled outside the
#' boundaries when `fill_mode="constant"`.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape `(batch, height, width, channels)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, channels, height, width)`. It defaults to the
#' `image_data_format` value found in your Keras config file at
#' `~/.keras/keras.json`. If you never set it, then it will be
#' `"channels_last"`.
#'
#' @param ...
#' Base layer keyword arguments, such as `name` and `dtype`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @inherit layer_dense return
#' @export
#' @family image augmentation layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_translation#randomtranslation-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomTranslation>
#' @tether keras.layers.RandomTranslation
layer_random_translation <-
function (object, height_factor, width_factor, fill_mode = "reflect",
    interpolation = "bilinear", seed = NULL, fill_value = 0,
    data_format = NULL, ...)
{
    args <- capture_args(list(seed = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$RandomTranslation, object, args)
}


#' A preprocessing layer which randomly zooms images during training.
#'
#' @description
#' This layer will randomly zoom in or out on each axis of an image
#' independently, filling empty space according to `fill_mode`.
#'
#' Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
#' of integer or floating point dtype.
#' By default, the layer will output floats.
#'
#' # Input Shape
#' 3D (unbatched) or 4D (batched) tensor with shape:
#' `(..., height, width, channels)`, in `"channels_last"` format,
#' or `(..., channels, height, width)`, in `"channels_first"` format.
#'
#' # Output Shape
#' 3D (unbatched) or 4D (batched) tensor with shape:
#'     `(..., target_height, target_width, channels)`,
#'     or `(..., channels, target_height, target_width)`,
#'     in `"channels_first"` format.
#'
#' **Note:** This layer is safe to use inside a `tf.data` pipeline
#' (independently of which backend you're using).
#'
#' # Examples
#' ```{r}
#' input_img <- random_uniform(c(32, 224, 224, 3))
#' layer <- layer_random_zoom(height_factor = .5, width_factor = .2)
#' out_img <- layer(input_img)
#' ```
#'
#' @param height_factor
#' a float represented as fraction of value, or a list of
#' size 2 representing lower and upper bound for zooming vertically.
#' When represented as a single float, this value is used for both the
#' upper and lower bound. A positive value means zooming out, while a
#' negative value means zooming in. For instance,
#' `height_factor=c(0.2, 0.3)` result in an output zoomed out by a
#' random amount in the range `[+20%, +30%]`.
#' `height_factor=c(-0.3, -0.2)` result in an output zoomed in by a
#' random amount in the range `[+20%, +30%]`.
#'
#' @param width_factor
#' a float represented as fraction of value, or a list of
#' size 2 representing lower and upper bound for zooming horizontally.
#' When represented as a single float, this value is used for both the
#' upper and lower bound. For instance, `width_factor=c(0.2, 0.3)`
#' result in an output zooming out between 20% to 30%.
#' `width_factor=c(-0.3, -0.2)` result in an output zooming in between
#' 20% to 30%. `NULL` means i.e., zooming vertical and horizontal
#' directions by preserving the aspect ratio. Defaults to `NULL`.
#'
#' @param fill_mode
#' Points outside the boundaries of the input are filled
#' according to the given mode. Available methods are `"constant"`,
#' `"nearest"`, `"wrap"` and `"reflect"`. Defaults to `"constant"`.
#' - `"reflect"`: `(d c b a | a b c d | d c b a)`
#'     The input is extended by reflecting about the edge of the last
#'     pixel.
#' - `"constant"`: `(k k k k | a b c d | k k k k)`
#'     The input is extended by filling all values beyond
#'     the edge with the same constant value k specified by
#'     `fill_value`.
#' - `"wrap"`: `(a b c d | a b c d | a b c d)`
#'     The input is extended by wrapping around to the opposite edge.
#' - `"nearest"`: `(a a a a | a b c d | d d d d)`
#'     The input is extended by the nearest pixel.
#' Note that when using torch backend, `"reflect"` is redirected to
#' `"mirror"` `(c d c b | a b c d | c b a b)` because torch does not
#' support `"reflect"`.
#' Note that torch backend does not support `"wrap"`.
#'
#' @param interpolation
#' Interpolation mode. Supported values: `"nearest"`,
#' `"bilinear"`.
#'
#' @param seed
#' Integer. Used to create a random seed.
#'
#' @param fill_value
#' a float represents the value to be filled outside
#' the boundaries when `fill_mode="constant"`.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape `(batch, height, width, channels)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, channels, height, width)`. It defaults to the
#' `image_data_format` value found in your Keras config file at
#' `~/.keras/keras.json`. If you never set it, then it will be
#' `"channels_last"`.
#'
#' @param ...
#' Base layer keyword arguments, such as `name` and `dtype`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @inherit layer_dense return
#' @export
#' @family image augmentation layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_zoom#randomzoom-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomZoom>
#'
#' @tether keras.layers.RandomZoom
layer_random_zoom <-
function (object, height_factor, width_factor = NULL, fill_mode = "reflect",
    interpolation = "bilinear", seed = NULL, fill_value = 0,
    data_format = NULL, ...)
{
    args <- capture_args(list(seed = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$RandomZoom, object, args)
}


#' A preprocessing layer which rescales input values to a new range.
#'
#' @description
#' This layer rescales every value of an input (often an image) by multiplying
#' by `scale` and adding `offset`.
#'
#' For instance:
#'
#' 1. To rescale an input in the `[0, 255]` range
#' to be in the `[0, 1]` range, you would pass `scale=1./255`.
#'
#' 2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` range,
#' you would pass `scale=1./127.5, offset=-1`.
#'
#' The rescaling is applied both during training and inference. Inputs can be
#' of integer or floating point dtype, and by default the layer will output
#' floats.
#'
#' **Note:** This layer is safe to use inside a `tf.data` pipeline
#' (independently of which backend you're using).
#'
#' @param scale
#' Float, the scale to apply to the inputs.
#'
#' @param offset
#' Float, the offset to apply to the inputs.
#'
#' @param ...
#' Base layer keyword arguments, such as `name` and `dtype`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @inherit layer_dense return
#' @export
#' @family image preprocessing layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/image_preprocessing/rescaling#rescaling-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling>
#' @tether keras.layers.Rescaling
layer_rescaling <-
function (object, scale, offset = 0, ...)
{
    args <- capture_args(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$Rescaling, object, args)
}


#' A preprocessing layer which resizes images.
#'
#' @description
#' This layer resizes an image input to a target height and width. The input
#' should be a 4D (batched) or 3D (unbatched) tensor in `"channels_last"`
#' format. Input pixel values can be of any range
#' (e.g. `[0., 1.)` or `[0, 255]`).
#'
#' # Input Shape
#' 3D (unbatched) or 4D (batched) tensor with shape:
#' `(..., height, width, channels)`, in `"channels_last"` format,
#' or `(..., channels, height, width)`, in `"channels_first"` format.
#'
#' # Output Shape
#' 3D (unbatched) or 4D (batched) tensor with shape:
#'     `(..., target_height, target_width, channels)`,
#'     or `(..., channels, target_height, target_width)`,
#'     in `"channels_first"` format.
#'
#' **Note:** This layer is safe to use inside a `tf.data` pipeline
#' (independently of which backend you're using).
#'
#' @param height
#' Integer, the height of the output shape.
#'
#' @param width
#' Integer, the width of the output shape.
#'
#' @param interpolation
#' String, the interpolation method.
#' Supports `"bilinear"`, `"nearest"`, `"bicubic"`,
#' `"lanczos3"`, `"lanczos5"`. Defaults to `"bilinear"`.
#'
#' @param crop_to_aspect_ratio
#' If `TRUE`, resize the images without aspect
#' ratio distortion. When the original aspect ratio differs
#' from the target aspect ratio, the output image will be
#' cropped so as to return the
#' largest possible window in the image (of size `(height, width)`)
#' that matches the target aspect ratio. By default
#' (`crop_to_aspect_ratio=FALSE`), aspect ratio may not be preserved.
#'
#' @param data_format
#' string, either `"channels_last"` or `"channels_first"`.
#' The ordering of the dimensions in the inputs. `"channels_last"`
#' corresponds to inputs with shape `(batch, height, width, channels)`
#' while `"channels_first"` corresponds to inputs with shape
#' `(batch, channels, height, width)`. It defaults to the
#' `image_data_format` value found in your Keras config file at
#' `~/.keras/keras.json`. If you never set it, then it will be
#' `"channels_last"`.
#'
#' @param ...
#' Base layer keyword arguments, such as `name` and `dtype`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @inherit layer_dense return
#' @export
#' @family image preprocessing layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/image_preprocessing/resizing#resizing-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Resizing>
#' @tether keras.layers.Resizing
layer_resizing <-
function (object, height, width, interpolation = "bilinear",
    crop_to_aspect_ratio = FALSE, data_format = NULL, ...)
{
    args <- capture_args(list(height = as_integer, width = as_integer,
        input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$Resizing, object, args)
}


#' A preprocessing layer that maps strings to (possibly encoded) indices.
#'
#' @description
#' This layer translates a set of arbitrary strings into integer output via a
#' table-based vocabulary lookup. This layer will perform no splitting or
#' transformation of input strings. For a layer than can split and tokenize
#' natural language, see the `layer_text_vectorization` layer.
#'
#' The vocabulary for the layer must be either supplied on construction or
#' learned via `adapt()`. During `adapt()`, the layer will analyze a data set,
#' determine the frequency of individual strings tokens, and create a
#' vocabulary from them. If the vocabulary is capped in size, the most frequent
#' tokens will be used to create the vocabulary and all others will be treated
#' as out-of-vocabulary (OOV).
#'
#' There are two possible output modes for the layer.
#' When `output_mode` is `"int"`,
#' input strings are converted to their index in the vocabulary (an integer).
#' When `output_mode` is `"multi_hot"`, `"count"`, or `"tf_idf"`, input strings
#' are encoded into an array where each dimension corresponds to an element in
#' the vocabulary.
#'
#' The vocabulary can optionally contain a mask token as well as an OOV token
#' (which can optionally occupy multiple indices in the vocabulary, as set
#' by `num_oov_indices`).
#' The position of these tokens in the vocabulary is fixed. When `output_mode`
#' is `"int"`, the vocabulary will begin with the mask token (if set), followed
#' by OOV indices, followed by the rest of the vocabulary. When `output_mode`
#' is `"multi_hot"`, `"count"`, or `"tf_idf"` the vocabulary will begin with
#' OOV indices and instances of the mask token will be dropped.
#'
#' **Note:** This layer uses TensorFlow internally. It cannot
#' be used as part of the compiled computation graph of a model with
#' any backend other than TensorFlow.
#' It can however be used with any backend when running eagerly.
#' It can also always be used as part of an input preprocessing pipeline
#' with any backend (outside the model itself), which is how we recommend
#' to use this layer.
#'
#' **Note:** This layer is safe to use inside a `tf.data` pipeline
#' (independently of which backend you're using).
#'
#' # Examples
#' **Creating a lookup layer with a known vocabulary**
#'
#' This example creates a lookup layer with a pre-existing vocabulary.
#'
#' ```{r}
#' vocab <- c("a", "b", "c", "d")
#' data <- rbind(c("a", "c", "d"), c("d", "z", "b"))
#' layer <- layer_string_lookup(vocabulary=vocab)
#' layer(data)
#' ```
#'
#' **Creating a lookup layer with an adapted vocabulary**
#'
#' This example creates a lookup layer and generates the vocabulary by
#' analyzing the dataset.
#'
#' ```{r}
#' data <- rbind(c("a", "c", "d"), c("d", "z", "b"))
#' layer <- layer_string_lookup()
#' layer %>% adapt(data)
#' get_vocabulary(layer)
#' ```
#'
#' Note that the OOV token `"[UNK]"` has been added to the vocabulary.
#' The remaining tokens are sorted by frequency
#' (`"d"`, which has 2 occurrences, is first) then by inverse sort order.
#'
#' ```{r}
#' data <- rbind(c("a", "c", "d"), c("d", "z", "b"))
#' layer <- layer_string_lookup()
#' layer %>% adapt(data)
#' layer(data)
#' ```
#'
#' **Lookups with multiple OOV indices**
#'
#' This example demonstrates how to use a lookup layer with multiple OOV
#' indices.  When a layer is created with more than one OOV index, any OOV
#' values are hashed into the number of OOV buckets, distributing OOV values in
#' a deterministic fashion across the set.
#'
#' ```{r}
#' vocab <- c("a", "b", "c", "d")
#' data <- rbind(c("a", "c", "d"), c("m", "z", "b"))
#' layer <- layer_string_lookup(vocabulary = vocab, num_oov_indices = 2)
#' layer(data)
#' ```
#'
#' Note that the output for OOV value 'm' is 0, while the output for OOV value
#' `"z"` is 1. The in-vocab terms have their output index increased by 1 from
#' earlier examples (a maps to 2, etc) in order to make space for the extra OOV
#' value.
#'
#' **One-hot output**
#'
#' Configure the layer with `output_mode='one_hot'`. Note that the first
#' `num_oov_indices` dimensions in the ont_hot encoding represent OOV values.
#'
#' ```{r}
#' vocab <- c("a", "b", "c", "d")
#' data <- c("a", "b", "c", "d", "z")
#' layer <- layer_string_lookup(vocabulary = vocab, output_mode = 'one_hot')
#' layer(data)
#' ```
#'
#' **Multi-hot output**
#'
#' Configure the layer with `output_mode='multi_hot'`. Note that the first
#' `num_oov_indices` dimensions in the multi_hot encoding represent OOV values.
#'
#' ```{r}
#' vocab <- c("a", "b", "c", "d")
#' data <- rbind(c("a", "c", "d", "d"), c("d", "z", "b", "z"))
#' layer <- layer_string_lookup(vocabulary = vocab, output_mode = 'multi_hot')
#' layer(data)
#' ```
#'
#' **Token count output**
#'
#' Configure the layer with `output_mode='count'`. As with multi_hot output,
#' the first `num_oov_indices` dimensions in the output represent OOV values.
#'
#' ```{r}
#' vocab <- c("a", "b", "c", "d")
#' data <- rbind(c("a", "c", "d", "d"), c("d", "z", "b", "z"))
#' layer <- layer_string_lookup(vocabulary = vocab, output_mode = 'count')
#' layer(data)
#' ```
#'
#' **TF-IDF output**
#'
#' Configure the layer with `output_mode="tf_idf"`. As with multi_hot output,
#' the first `num_oov_indices` dimensions in the output represent OOV values.
#'
#' Each token bin will output `token_count * idf_weight`, where the idf weights
#' are the inverse document frequency weights per token. These should be
#' provided along with the vocabulary. Note that the `idf_weight` for OOV
#' values will default to the average of all idf weights passed in.
#'
#' ```{r}
#' vocab <- c("a", "b", "c", "d")
#' idf_weights <- c(0.25, 0.75, 0.6, 0.4)
#' data <- rbind(c("a", "c", "d", "d"), c("d", "z", "b", "z"))
#' layer <- layer_string_lookup(output_mode = "tf_idf")
#' layer %>% set_vocabulary(vocab, idf_weights=idf_weights)
#' layer(data)
#' ```
#'
#' To specify the idf weights for oov values, you will need to pass the entire
#' vocabulary including the leading oov token.
#'
#' ```{r}
#' vocab <- c("[UNK]", "a", "b", "c", "d")
#' idf_weights <- c(0.9, 0.25, 0.75, 0.6, 0.4)
#' data <- rbind(c("a", "c", "d", "d"), c("d", "z", "b", "z"))
#' layer <- layer_string_lookup(output_mode = "tf_idf")
#' layer %>% set_vocabulary(vocab, idf_weights=idf_weights)
#' layer(data)
#' ```
#'
#' When adapting the layer in `"tf_idf"` mode, each input sample will be
#' considered a document, and IDF weight per token will be calculated as
#' `log(1 + num_documents / (1 + token_document_count))`.
#'
#' **Inverse lookup**
#'
#' This example demonstrates how to map indices to strings using this layer.
#' (You can also use `adapt()` with `inverse=TRUE`, but for simplicity we'll
#' pass the vocab in this example.)
#'
#' ```{r}
#' vocab <- c("a", "b", "c", "d")
#' data <- rbind(c(1, 3, 4), c(4, 0, 2))
#' layer <- layer_string_lookup(vocabulary = vocab, invert = TRUE)
#' layer(data)
#' ```
#'
#' Note that the first index correspond to the oov token by default.
#'
#' **Forward and inverse lookup pairs**
#'
#' This example demonstrates how to use the vocabulary of a standard lookup
#' layer to create an inverse lookup layer.
#'
#' ```{r}
#' vocab <- c("a", "b", "c", "d")
#' data <- rbind(c("a", "c", "d"), c("d", "z", "b"))
#' layer <- layer_string_lookup(vocabulary = vocab)
#' i_layer <- layer_string_lookup(vocabulary = vocab, invert = TRUE)
#' int_data <- layer(data)
#' i_layer(int_data)
#' ```
#'
#' In this example, the input value `"z"` resulted in an output of `"[UNK]"`,
#' since 1000 was not in the vocabulary - it got represented as an OOV, and all
#' OOV values are returned as `"[UNK]"` in the inverse layer. Also, note that
#' for the inverse to work, you must have already set the forward layer
#' vocabulary either directly or via `adapt()` before calling
#' `get_vocabulary()`.
#'
#' @param max_tokens
#' Maximum size of the vocabulary for this layer. This should
#' only be specified when adapting the vocabulary or when setting
#' `pad_to_max_tokens=TRUE`. If NULL, there is no cap on the size of
#' the vocabulary. Note that this size includes the OOV
#' and mask tokens. Defaults to `NULL`.
#'
#' @param num_oov_indices
#' The number of out-of-vocabulary tokens to use.
#' If this value is more than 1, OOV inputs are modulated to
#' determine their OOV value.
#' If this value is 0, OOV inputs will cause an error when calling
#' the layer. Defaults to `1`.
#'
#' @param mask_token
#' A token that represents masked inputs. When `output_mode` is
#' `"int"`, the token is included in vocabulary and mapped to index 0.
#' In other output modes, the token will not appear
#' in the vocabulary and instances of the mask token
#' in the input will be dropped. If set to `NULL`,
#' no mask term will be added. Defaults to `NULL`.
#'
#' @param oov_token
#' Only used when `invert` is TRUE. The token to return for OOV
#' indices. Defaults to `"[UNK]"`.
#'
#' @param vocabulary
#' Optional. Either an array of integers or a string path to a
#' text file. If passing an array, can pass a list, list,
#' 1D NumPy array, or 1D tensor containing the integer vocbulary terms.
#' If passing a file path, the file should contain one line per term
#' in the vocabulary. If this argument is set,
#' there is no need to `adapt()` the layer.
#'
#' @param vocabulary_dtype
#' The dtype of the vocabulary terms, for example
#' `"int64"` or `"int32"`. Defaults to `"int64"`.
#'
#' @param idf_weights
#' Only valid when `output_mode` is `"tf_idf"`.
#' A list, list, 1D NumPy array, or 1D tensor or the same length
#' as the vocabulary, containing the floating point inverse document
#' frequency weights, which will be multiplied by per sample term
#' counts for the final TF-IDF weight.
#' If the `vocabulary` argument is set, and `output_mode` is
#' `"tf_idf"`, this argument must be supplied.
#'
#' @param invert
#' Only valid when `output_mode` is `"int"`.
#' If `TRUE`, this layer will map indices to vocabulary items
#' instead of mapping vocabulary items to indices.
#' Defaults to `FALSE`.
#'
#' @param output_mode
#' Specification for the output of the layer. Values can be
#' `"int"`, `"one_hot"`, `"multi_hot"`, `"count"`, or `"tf_idf"`
#' configuring the layer as follows:
#' - `"int"`: Return the vocabulary indices of the input tokens.
#' - `"one_hot"`: Encodes each individual element in the input into an
#'     array the same size as the vocabulary,
#'     containing a 1 at the element index. If the last dimension
#'     is size 1, will encode on that dimension.
#'     If the last dimension is not size 1, will append a new
#'     dimension for the encoded output.
#' - `"multi_hot"`: Encodes each sample in the input into a single
#'     array the same size as the vocabulary,
#'     containing a 1 for each vocabulary term present in the sample.
#'     Treats the last dimension as the sample dimension,
#'     if input shape is `(..., sample_length)`,
#'     output shape will be `(..., num_tokens)`.
#' - `"count"`: As `"multi_hot"`, but the int array contains
#'     a count of the number of times the token at that index
#'     appeared in the sample.
#' - `"tf_idf"`: As `"multi_hot"`, but the TF-IDF algorithm is
#'     applied to find the value in each token slot.
#' For `"int"` output, any shape of input and output is supported.
#' For all other output modes, currently only output up to rank 2
#' is supported. Defaults to `"int"`.
#'
#' @param pad_to_max_tokens
#' Only applicable when `output_mode` is `"multi_hot"`,
#' `"count"`, or `"tf_idf"`. If `TRUE`, the output will have
#' its feature axis padded to `max_tokens` even if the number
#' of unique tokens in the vocabulary is less than `max_tokens`,
#' resulting in a tensor of shape `(batch_size, max_tokens)`
#' regardless of vocabulary size. Defaults to `FALSE`.
#'
#' @param sparse
#' Boolean. Only applicable to `"multi_hot"`, `"count"`, and
#' `"tf_idf"` output modes. Only supported with TensorFlow
#' backend. If `TRUE`, returns a `SparseTensor`
#' instead of a dense `Tensor`. Defaults to `FALSE`.
#'
#' @param encoding
#' Optional. The text encoding to use to interpret the input
#' strings. Defaults to `"utf-8"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param name
#' String, name for the object
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit layer_dense return
#' @export
#' @family categorical features preprocessing layers
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/categorical/string_lookup#stringlookup-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/StringLookup>
#'
#' @tether keras.layers.StringLookup
layer_string_lookup <-
function (object, max_tokens = NULL, num_oov_indices = 1L, mask_token = NULL,
    oov_token = "[UNK]", vocabulary = NULL, idf_weights = NULL,
    invert = FALSE, output_mode = "int", pad_to_max_tokens = FALSE,
    sparse = FALSE, encoding = "utf-8", name = NULL, ..., vocabulary_dtype = NULL)
{
    args <- capture_args(list(num_oov_indices = as_integer,
        mask_token = as_integer, vocabulary = as_integer, invert = as_integer,
        output_mode = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$StringLookup, object, args)
}


#' A preprocessing layer which maps text features to integer sequences.
#'
#' @description
#' This layer has basic options for managing text in a Keras model. It
#' transforms a batch of strings (one example = one string) into either a list
#' of token indices (one example = 1D tensor of integer token indices) or a
#' dense representation (one example = 1D tensor of float values representing
#' data about the example's tokens). This layer is meant to handle natural
#' language inputs. To handle simple string inputs (categorical strings or
#' pre-tokenized strings) see `layer_string_lookup()`.
#'
#' The vocabulary for the layer must be either supplied on construction or
#' learned via `adapt()`. When this layer is adapted, it will analyze the
#' dataset, determine the frequency of individual string values, and create a
#' vocabulary from them. This vocabulary can have unlimited size or be capped,
#' depending on the configuration options for this layer; if there are more
#' unique values in the input than the maximum vocabulary size, the most
#' frequent terms will be used to create the vocabulary.
#'
#' The processing of each example contains the following steps:
#'
#' 1. Standardize each example (usually lowercasing + punctuation stripping)
#' 2. Split each example into substrings (usually words)
#' 3. Recombine substrings into tokens (usually ngrams)
#' 4. Index tokens (associate a unique int value with each token)
#' 5. Transform each example using this index, either into a vector of ints or
#'    a dense float vector.
#'
#' Some notes on passing callables to customize splitting and normalization for
#' this layer:
#'
#' 1. Any callable can be passed to this Layer, but if you want to serialize
#'    this object you should only pass functions that are registered Keras
#'    serializables (see [`register_keras_serializable()`]
#'    for more details).
#' 2. When using a custom callable for `standardize`, the data received
#'    by the callable will be exactly as passed to this layer. The callable
#'    should return a tensor of the same shape as the input.
#' 3. When using a custom callable for `split`, the data received by the
#'    callable will have the 1st dimension squeezed out - instead of
#'    `list("string to split", "another string to split")`, the Callable will
#'    see `c("string to split", "another string to split")`.
#'    The callable should return a `tf.Tensor` of dtype `string`
#'    with the first dimension containing the split tokens -
#'    in this example, we should see something like `list(c("string", "to",
#'    "split"), c("another", "string", "to", "split"))`.
#'
#' **Note:** This layer uses TensorFlow internally. It cannot
#' be used as part of the compiled computation graph of a model with
#' any backend other than TensorFlow.
#' It can however be used with any backend when running eagerly.
#' It can also always be used as part of an input preprocessing pipeline
#' with any backend (outside the model itself), which is how we recommend
#' to use this layer.
#'
#' **Note:** This layer is safe to use inside a `tf.data` pipeline
#' (independently of which backend you're using).
#'
#' # Examples
#' This example instantiates a `TextVectorization` layer that lowercases text,
#' splits on whitespace, strips punctuation, and outputs integer vocab indices.
#'
#' ```{r}
#' max_tokens <- 5000  # Maximum vocab size.
#' max_len <- 4  # Sequence length to pad the outputs to.
#' # Create the layer.
#' vectorize_layer <- layer_text_vectorization(
#'     max_tokens = max_tokens,
#'     output_mode = 'int',
#'     output_sequence_length = max_len)
#' ```
#'
#' ```{r}
#' # Now that the vocab layer has been created, call `adapt` on the
#' # list of strings to create the vocabulary.
#' vectorize_layer %>% adapt(c("foo bar", "bar baz", "baz bada boom"))
#' ```
#'
#' ```{r}
#' # Now, the layer can map strings to integers -- you can use an
#' # embedding layer to map these integers to learned embeddings.
#' input_data <- rbind("foo qux bar", "qux baz")
#' vectorize_layer(input_data)
#' ```
#'
#' This example instantiates a `TextVectorization` layer by passing a list
#' of vocabulary terms to the layer's `initialize()` method.
#'
#' ```{r}
#' vocab_data <- c("earth", "wind", "and", "fire")
#' max_len <- 4  # Sequence length to pad the outputs to.
#' # Create the layer, passing the vocab directly. You can also pass the
#' # vocabulary arg a path to a file containing one vocabulary word per
#' # line.
#' vectorize_layer <- layer_text_vectorization(
#'     max_tokens = max_tokens,
#'     output_mode = 'int',
#'     output_sequence_length = max_len,
#'     vocabulary = vocab_data)
#' ```
#'
#' ```{r}
#' # Because we've passed the vocabulary directly, we don't need to adapt
#' # the layer - the vocabulary is already set. The vocabulary contains the
#' # padding token ('') and OOV token ('[UNK]')
#' # as well as the passed tokens.
#' vectorize_layer %>% get_vocabulary()
#' # ['', '[UNK]', 'earth', 'wind', 'and', 'fire']
#' ```
#'
#' @param max_tokens
#' Maximum size of the vocabulary for this layer. This should
#' only be specified when adapting a vocabulary or when setting
#' `pad_to_max_tokens=TRUE`. Note that this vocabulary
#' contains 1 OOV token, so the effective number of tokens is
#' `(max_tokens - 1 - (1 if output_mode == "int" else 0))`.
#'
#' @param standardize
#' Optional specification for standardization to apply to the
#' input text. Values can be:
#' - `NULL`: No standardization.
#' - `"lower_and_strip_punctuation"`: Text will be lowercased and all
#'     punctuation removed.
#' - `"lower"`: Text will be lowercased.
#' - `"strip_punctuation"`: All punctuation will be removed.
#' - Callable: Inputs will passed to the callable function,
#'     which should be standardized and returned.
#'
#' @param split
#' Optional specification for splitting the input text.
#' Values can be:
#' - `NULL`: No splitting.
#' - `"whitespace"`: Split on whitespace.
#' - `"character"`: Split on each unicode character.
#' - Callable: Standardized inputs will passed to the callable
#'     function, which should be split and returned.
#'
#' @param ngrams
#' Optional specification for ngrams to create from the
#' possibly-split input text. Values can be `NULL`, an integer
#' or list of integers; passing an integer will create ngrams
#' up to that integer, and passing a list of integers will
#' create ngrams for the specified values in the list.
#' Passing `NULL` means that no ngrams will be created.
#'
#' @param output_mode
#' Optional specification for the output of the layer.
#' Values can be `"int"`, `"multi_hot"`, `"count"` or `"tf_idf"`,
#' configuring the layer as follows:
#' - `"int"`: Outputs integer indices, one integer index per split
#'     string token. When `output_mode == "int"`,
#'     0 is reserved for masked locations;
#'     this reduces the vocab size to `max_tokens - 2`
#'     instead of `max_tokens - 1`.
#' - `"multi_hot"`: Outputs a single int array per batch, of either
#'     vocab_size or max_tokens size, containing 1s in all elements
#'     where the token mapped to that index exists at least
#'     once in the batch item.
#' - `"count"`: Like `"multi_hot"`, but the int array contains
#'     a count of the number of times the token at that index
#'     appeared in the batch item.
#' - `"tf_idf"`: Like `"multi_hot"`, but the TF-IDF algorithm
#'     is applied to find the value in each token slot.
#' For `"int"` output, any shape of input and output is supported.
#' For all other output modes, currently only rank 1 inputs
#' (and rank 2 outputs after splitting) are supported.
#'
#' @param output_sequence_length
#' Only valid in INT mode. If set, the output will
#' have its time dimension padded or truncated to exactly
#' `output_sequence_length` values, resulting in a tensor of shape
#' `(batch_size, output_sequence_length)` regardless of how many tokens
#' resulted from the splitting step. Defaults to `NULL`. If `ragged`
#' is `TRUE` then `output_sequence_length` may still truncate the
#' output.
#'
#' @param pad_to_max_tokens
#' Only valid in  `"multi_hot"`, `"count"`,
#' and `"tf_idf"` modes. If `TRUE`, the output will have
#' its feature axis padded to `max_tokens` even if the number
#' of unique tokens in the vocabulary is less than `max_tokens`,
#' resulting in a tensor of shape `(batch_size, max_tokens)`
#' regardless of vocabulary size. Defaults to `FALSE`.
#'
#' @param vocabulary
#' Optional. Either an array of strings or a string path to a
#' text file. If passing an array, can pass a list, list,
#' 1D NumPy array, or 1D tensor containing the string vocabulary terms.
#' If passing a file path, the file should contain one line per term
#' in the vocabulary. If this argument is set,
#' there is no need to `adapt()` the layer.
#'
#' @param idf_weights
#' Only valid when `output_mode` is `"tf_idf"`. A list, list,
#' 1D NumPy array, or 1D tensor of the same length as the vocabulary,
#' containing the floating point inverse document frequency weights,
#' which will be multiplied by per sample term counts for
#' the final `tf_idf` weight. If the `vocabulary` argument is set,
#' and `output_mode` is `"tf_idf"`, this argument must be supplied.
#'
#' @param ragged
#' Boolean. Only applicable to `"int"` output mode.
#' Only supported with TensorFlow backend.
#' If `TRUE`, returns a `RaggedTensor` instead of a dense `Tensor`,
#' where each sequence may have a different length
#' after string splitting. Defaults to `FALSE`.
#'
#' @param sparse
#' Boolean. Only applicable to `"multi_hot"`, `"count"`, and
#' `"tf_idf"` output modes. Only supported with TensorFlow
#' backend. If `TRUE`, returns a `SparseTensor`
#' instead of a dense `Tensor`. Defaults to `FALSE`.
#'
#' @param encoding
#' Optional. The text encoding to use to interpret the input
#' strings. Defaults to `"utf-8"`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param name
#' String, name for the object
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit layer_dense return
#' @export
#' @family preprocessing layers
#' @family layers
#' @seealso
#' + <https://keras.io/api/layers/preprocessing_layers/text/text_vectorization#textvectorization-class>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization>
#'
#' @tether keras.layers.TextVectorization
layer_text_vectorization <-
function (object, max_tokens = NULL, standardize = "lower_and_strip_punctuation",
    split = "whitespace", ngrams = NULL, output_mode = "int",
    output_sequence_length = NULL, pad_to_max_tokens = FALSE,
    vocabulary = NULL, idf_weights = NULL, sparse = FALSE, ragged = FALSE,
    encoding = "utf-8", name = NULL, ...)
{
    args <- capture_args(list(max_tokens = as_integer, ngrams = function (x)
    if (length(x) > 1)
        as_integer_tuple(x)
    else as_integer(x), output_mode = as_integer, output_sequence_length = as_integer,
        ragged = as_integer, input_shape = normalize_shape, batch_size = as_integer,
        batch_input_shape = normalize_shape), ignore = "object")
    create_layer(keras$layers$TextVectorization, object, args)
}



# TODO: add tests/ confirm that `get_vocabulary()` returns an R character
# vector. In older TF versions it used to return python byte objects, which
# needed `x.decode("UTF-8") for x in vocab]`

#' @param include_special_tokens If TRUE, the returned vocabulary will include
#'   the padding and OOV tokens, and a term's index in the vocabulary will equal
#'   the term's index when calling the layer. If FALSE, the returned vocabulary
#'   will not include any padding or OOV tokens.
#' @rdname layer_text_vectorization
#' @export
get_vocabulary <- function(object, include_special_tokens=TRUE) {
  args <- capture_args(ignore = "object")
  do.call(object$get_vocabulary, args)
}

#' @rdname layer_text_vectorization
#' @param idf_weights An R vector, 1D numpy array, or 1D tensor of inverse
#'   document frequency weights with equal length to vocabulary. Must be set if
#'   output_mode is "tf_idf". Should not be set otherwise.
#' @export
set_vocabulary <- function(object, vocabulary, idf_weights=NULL, ...) {
  args <- capture_args(ignore = "object")
  do.call(object$set_vocabulary, args)
  invisible(object)
}


## TODO: TextVectorization has a compile() method. investigate if this is
## actually useful to export
#compile.keras.engine.base_preprocessing_layer.PreprocessingLayer <-
function(object, run_eagerly = NULL, steps_per_execution = NULL, ...) {
  args <- capture_args(ignore="object")
  do.call(object$compile, args)
}


#' A preprocessing layer to convert raw audio signals to Mel spectrograms.
#'
#' @description
#' This layer takes `float32`/`float64` single or batched audio signal as
#' inputs and computes the Mel spectrogram using Short-Time Fourier Transform
#' and Mel scaling. The input should be a 1D (unbatched) or 2D (batched) tensor
#' representing audio signals. The output will be a 2D or 3D tensor
#' representing Mel spectrograms.
#'
#' A spectrogram is an image-like representation that shows the frequency
#' spectrum of a signal over time. It uses x-axis to represent time, y-axis to
#' represent frequency, and each pixel to represent intensity.
#' Mel spectrograms are a special type of spectrogram that use the mel scale,
#' which approximates how humans perceive sound. They are commonly used in
#' speech and music processing tasks like speech recognition, speaker
#' identification, and music genre classification.
#'
#' # References
#' - [Spectrogram](https://en.wikipedia.org/wiki/Spectrogram),
#' - [Mel scale](https://en.wikipedia.org/wiki/Mel_scale).
#'
#' # Examples
#' **Unbatched audio signal**
#'
#' ```r
#' layer <- layer_mel_spectrogram(
#'   num_mel_bins = 64,
#'   sampling_rate = 8000,
#'   sequence_stride = 256,
#'   fft_length = 2048
#' )
#' layer(random_uniform(shape = c(16000))) |> shape()
#' ```
#'
#' **Batched audio signal**
#'
#' ```r
#' layer <- layer_mel_spectrogram(
#'   num_mel_bins = 80,
#'   sampling_rate = 8000,
#'   sequence_stride = 128,
#'   fft_length = 2048
#' )
#' layer(random_uniform(shape = c(2, 16000))) |> shape()
#' ```
#'
#' # Input Shape
#' 1D (unbatched) or 2D (batched) tensor with shape:`(..., samples)`.
#'
#' # Output Shape
#' 2D (unbatched) or 3D (batched) tensor with
#' shape:`(..., num_mel_bins, time)`.
#'
#' @param fft_length
#' Integer, size of the FFT window.
#'
#' @param sequence_stride
#' Integer, number of samples between successive STFT
#' columns.
#'
#' @param sequence_length
#' Integer, size of the window used for applying
#' `window` to each audio frame. If `NULL`, defaults to `fft_length`.
#'
#' @param window
#' String, name of the window function to use. Available values
#' are `"hann"` and `"hamming"`. If `window` is a tensor, it will be
#' used directly as the window and its length must be
#' `sequence_length`. If `window` is `NULL`, no windowing is
#' used. Defaults to `"hann"`.
#'
#' @param sampling_rate
#' Integer, sample rate of the input signal.
#'
#' @param num_mel_bins
#' Integer, number of mel bins to generate.
#'
#' @param min_freq
#' Float, minimum frequency of the mel bins.
#'
#' @param max_freq
#' Float, maximum frequency of the mel bins.
#' If `NULL`, defaults to `sampling_rate / 2`.
#'
#' @param power_to_db
#' If TRUE, convert the power spectrogram to decibels.
#'
#' @param top_db
#' Float, minimum negative cut-off `max(10 * log10(S)) - top_db`.
#'
#' @param mag_exp
#' Float, exponent for the magnitude spectrogram.
#' 1 for magnitude, 2 for power, etc. Default is 2.
#'
#' @param ref_power
#' Float, the power is scaled relative to it
#' `10 * log10(S / ref_power)`.
#'
#' @param min_power
#' Float, minimum value for power and `ref_power`.
#'
#' @param object
#' Object to compose the layer with. A tensor, array, or sequential model.
#'
#' @param ...
#' For forward/backward compatability.
#'
#' @inherit layer_dense return
#' @family audio preprocessing layers
#' @family preprocessing layers
#' @family layers
#' @export
#' @tether keras.layers.MelSpectrogram
layer_mel_spectrogram <-
function (object, fft_length = 2048L, sequence_stride = 512L,
    sequence_length = NULL, window = "hann", sampling_rate = 16000L,
    num_mel_bins = 128L, min_freq = 20, max_freq = NULL, power_to_db = TRUE,
    top_db = 80, mag_exp = 2, min_power = 1e-10, ref_power = 1,
    ...)
{
    args <- capture_args(list(fft_length = as_integer, sequence_stride = as_integer,
        sequence_length = as_integer, sampling_rate = as_integer,
        num_mel_bins = as_integer, input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$MelSpectrogram, object, args)
}




# ---- adapt ----


#' Fits the state of the preprocessing layer to the data being passed
#'
#' @details
#' After calling `adapt` on a layer, a preprocessing layer's state will not
#' update during training. In order to make preprocessing layers efficient in
#' any distribution context, they are kept constant with respect to any
#' compiled `tf.Graph`s that call the layer. This does not affect the layer use
#' when adapting each layer only once, but if you adapt a layer multiple times
#' you will need to take care to re-compile any compiled functions as follows:
#'
#'  * If you are adding a preprocessing layer to a keras model, you need to
#'    call `compile(model)` after each subsequent call to `adapt()`.
#'  * If you are calling a preprocessing layer inside [`tfdatasets::dataset_map()`],
#'    you should call `dataset_map()` again on the input `Dataset` after each
#'    `adapt()`.
#'  * If you are using a [`tensorflow::tf_function()`] directly which calls a preprocessing
#'    layer, you need to call `tf_function()` again on your callable after
#'    each subsequent call to `adapt()`.
#'
#' `keras_model()` example with multiple adapts:
#' ````{r}
#' layer <- layer_normalization(axis = NULL)
#' adapt(layer, c(0, 2))
#' model <- keras_model_sequential() |> layer()
#' predict(model, c(0, 1, 2), verbose = FALSE) # [1] -1  0  1
#'
#' adapt(layer, c(-1, 1))
#' compile(model)  # This is needed to re-compile model.predict!
#' predict(model, c(0, 1, 2), verbose = FALSE) # [1] 0 1 2
#' ````
#'
#' `tfdatasets` example with multiple adapts:
#' ````{r}
#' layer <- layer_normalization(axis = NULL)
#' adapt(layer, c(0, 2))
#' input_ds <- tfdatasets::range_dataset(0, 3)
#' normalized_ds <- input_ds |>
#'   tfdatasets::dataset_map(layer)
#' str(tfdatasets::iterate(normalized_ds))
#'
#' adapt(layer, c(-1, 1))
#' normalized_ds <- input_ds |>
#'   tfdatasets::dataset_map(layer) # Re-map over the input dataset.
#'
#' normalized_ds |>
#'   tfdatasets::as_array_iterator() |>
#'   tfdatasets::iterate(simplify = FALSE) |>
#'   str()
#' ````
#'
#' @param object Preprocessing layer object
#'
#' @param data The data to train on. It can be passed either as a
#'   `tf.data.Dataset` or as an R array.
#'
#' @param batch_size Integer or `NULL`. Number of asamples per state update. If
#'   unspecified, `batch_size` will default to `32`. Do not specify the
#'   batch_size if your data is in the form of a TF Dataset or a generator
#'   (since they generate batches).
#'
#' @param steps Integer or `NULL`. Total number of steps (batches of samples)
#'   When training with input tensors such as TensorFlow data tensors, the
#'   default `NULL` is equal to the number of samples in your dataset divided by
#'   the batch size, or `1` if that cannot be determined. If x is a
#'   `tf.data.Dataset`, and `steps` is `NULL`, the epoch will run until the
#'   input dataset is exhausted. When passing an infinitely repeating dataset,
#'   you must specify the steps argument. This argument is not supported with
#'   array inputs.
#'
#' @param ... Used for forwards and backwards compatibility. Passed on to the underlying method.
#'
#' @family preprocessing layer methods
#'
# @seealso
#  + <https://www.tensorflow.org/guide/keras/preprocessing_layers#the_adapt_method>
#  + <https://keras.io/guides/preprocessing_layers/#the-adapt-method>
#'
#' @returns Returns `object`, invisibly.
#' @export
adapt <- function(object, data, ..., batch_size=NULL, steps=NULL) {
  if (!is_py_object(data))
    data <- keras_array(data)
  # TODO: use as_tensor() here

  args <- capture_args(list(batch_size = as_nullable_integer,
                             step = as_nullable_integer),
                        ignore = c("object", "data"))
  # `data` named to `dataset` in keras3 keras.utils.FeatureSpace
  # pass it as a positional arg
  args <- c(list(data), args)
  do.call(object$adapt, args)
  invisible(object)
}
