

#' Packs user-provided data into a list.
#'
#' @description
#' This is a convenience utility for packing data into the list formats
#' that `fit()` uses.
#'
#' # Example
#'
#' ```{r}
#' x <- op_ones(c(10, 1))
#' data <- pack_x_y_sample_weight(x)
#'
#'
#' y <- op_ones(c(10, 1))
#' data <- pack_x_y_sample_weight(x, y)
#' ```
#'
#' @returns
#' List in the format used in `fit()`.
#'
#' @param x
#' Features to pass to `Model`.
#'
#' @param y
#' Ground-truth targets to pass to `Model`.
#'
#' @param sample_weight
#' Sample weight for each element.
#'
#' @noRd
# @export
#' @family data utils
#' @family utils
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/pack_x_y_sample_weight>
#'
#' @tether keras.utils.pack_x_y_sample_weight
#' @keywords internal
pack_x_y_sample_weight <-
function (x, y = NULL, sample_weight = NULL)
{
  args <- capture_args()
  do.call(keras$utils$pack_x_y_sample_weight, args)
}


#' Unpacks user-provided data list.
#'
#' @description
#' This is a convenience utility to be used when overriding
#' `$train_step`, `$test_step`, or `$predict_step`.
#' This utility makes it easy to support data of the form `(x,)`,
#' `(x, y)`, or `(x, y, sample_weight)`.
#'
#' # Example:
#'
#' ```{r}
#' features_batch <- op_ones(c(10, 5))
#' labels_batch <- op_zeros(c(10, 5))
#' data <- list(features_batch, labels_batch)
#' # `y` and `sample_weight` will default to `NULL` if not provided.
#' c(x, y, sample_weight) %<-% unpack_x_y_sample_weight(data)
#' ```
#'
#' You can also do the equivalent by providing default values to `%<-%`
#'
#' ```r
#' c(x, y = NULL, sample_weight = NULL) %<-% data
#' ```
#' @returns
#' The unpacked list, with `NULL`s for `y` and `sample_weight` if they are
#' not provided.
#'
#' @param data
#' A list of the form `(x)`, `(x, y)`, or `(x, y, sample_weight)`.
#'
#' @noRd
# This is removed because it has no real purpose in R. Use this instead:
# c(x, y = NULL, sample_weight = NULL) %<-% data
#  @export
#' @family data utils
#' @family utils
#' @keywords internal
# @seealso
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/unpack_x_y_sample_weight>
#'
#' @tether keras.utils.unpack_x_y_sample_weight
unpack_x_y_sample_weight <-
function (data)
{
  args <- capture_args()
  do.call(keras$utils$unpack_x_y_sample_weight, args)
}


#' Generates a `tf.data.Dataset` from audio files in a directory.
#'
#' @description
#' If your directory structure is:
#'
#' ```
#' main_directory/
#' ...class_a/
#' ......a_audio_1.wav
#' ......a_audio_2.wav
#' ...class_b/
#' ......b_audio_1.wav
#' ......b_audio_2.wav
#' ```
#'
#' Then calling `audio_dataset_from_directory(main_directory,
#' labels = 'inferred')`
#' will return a `tf.data.Dataset` that yields batches of audio files from
#' the subdirectories `class_a` and `class_b`, together with labels
#' 0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).
#'
#' Only `.wav` files are supported at this time.
#'
#' @returns
#' A `tf.data.Dataset` object.
#'
#' - If `label_mode` is `NULL`, it yields `string` tensors of shape
#'   `(batch_size,)`, containing the contents of a batch of audio files.
#' - Otherwise, it yields a tuple `(audio, labels)`, where `audio`
#'   has shape `(batch_size, sequence_length, num_channels)` and `labels`
#'   follows the format described
#'   below.
#'
#' Rules regarding labels format:
#'
#' - if `label_mode` is `int`, the labels are an `int32` tensor of shape
#'   `(batch_size,)`.
#' - if `label_mode` is `binary`, the labels are a `float32` tensor of
#'   1s and 0s of shape `(batch_size, 1)`.
#' - if `label_mode` is `categorical`, the labels are a `float32` tensor
#'   of shape `(batch_size, num_classes)`, representing a one-hot
#'   encoding of the class index.
#'
#' @param directory
#' Directory where the data is located.
#' If `labels` is `"inferred"`, it should contain subdirectories,
#' each containing audio files for a class. Otherwise, the directory
#' structure is ignored.
#'
#' @param labels
#' Either "inferred" (labels are generated from the directory
#' structure), `NULL` (no labels), or a list/tuple of integer labels
#' of the same size as the number of audio files found in
#' the directory. Labels should be sorted according to the
#' alphanumeric order of the audio file paths
#' (obtained via `os.walk(directory)` in Python).
#'
#' @param label_mode
#' String describing the encoding of `labels`. Options are:
#' - `"int"`: means that the labels are encoded as integers (e.g. for
#'   `sparse_categorical_crossentropy` loss).
#' - `"categorical"` means that the labels are encoded as a categorical
#'   vector (e.g. for `categorical_crossentropy` loss)
#' - `"binary"` means that the labels (there can be only 2)
#'   are encoded as `float32` scalars with values 0
#'   or 1 (e.g. for `binary_crossentropy`).
#' - `NULL` (no labels).
#'
#' @param class_names
#' Only valid if "labels" is `"inferred"`.
#' This is the explicit list of class names
#' (must match names of subdirectories). Used to control the order
#' of the classes (otherwise alphanumerical order is used).
#'
#' @param batch_size
#' Size of the batches of data. Default: 32. If `NULL`,
#' the data will not be batched
#' (the dataset will yield individual samples).
#'
#' @param sampling_rate
#' Audio sampling rate (in samples per second).
#'
#' @param output_sequence_length
#' Maximum length of an audio sequence. Audio files
#' longer than this will be truncated to `output_sequence_length`.
#' If set to `NULL`, then all sequences in the same batch will
#' be padded to the
#' length of the longest sequence in the batch.
#'
#' @param ragged
#' Whether to return a Ragged dataset (where each sequence has its
#' own length). Defaults to `FALSE`.
#'
#' @param shuffle
#' Whether to shuffle the data. Defaults to `TRUE`.
#' If set to `FALSE`, sorts the data in alphanumeric order.
#'
#' @param seed
#' Optional random seed for shuffling and transformations.
#'
#' @param validation_split
#' Optional float between 0 and 1, fraction of data to
#' reserve for validation.
#'
#' @param subset
#' Subset of the data to return. One of `"training"`,
#' `"validation"` or `"both"`. Only used if `validation_split` is set.
#'
#' @param follow_links
#' Whether to visits subdirectories pointed to by symlinks.
#' Defaults to `FALSE`.
#'
#' @param verbose
#' Whether to display number information on classes and
#' number of files found. Defaults to `TRUE`.
#'
#'
#' @export
#' @family dataset utils
#' @family utils
#' @seealso
#' + <https://keras.io/api/data_loading/audio#audiodatasetfromdirectory-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/audio_dataset_from_directory>
#' @tether keras.utils.audio_dataset_from_directory
audio_dataset_from_directory <-
function (directory, labels = "inferred", label_mode = "int",
          class_names = NULL, batch_size = 32L, sampling_rate = NULL,
          output_sequence_length = NULL, ragged = FALSE, shuffle = TRUE,
          seed = NULL, validation_split = NULL, subset = NULL, follow_links = FALSE,
          verbose = TRUE)
{
  args <- capture_args(list(labels = as_integer, label_mode = as_integer,
                             batch_size = as_integer, seed = as_integer))
  do.call(keras$utils$audio_dataset_from_directory, args)
}


#' Splits a dataset into a left half and a right half (e.g. train / test).
#'
#' @description
#'
#' # Examples
#' ```{r}
#' data <- random_uniform(c(1000, 4))
#' c(left_ds, right_ds) %<-% split_dataset(list(data$numpy()), left_size = 0.8)
#' left_ds$cardinality()
#' right_ds$cardinality()
#' ```
#'
#' @returns
#' A list of two `tf$data$Dataset` objects:
#' the left and right splits.
#'
#' @param dataset
#' A `tf$data$Dataset`, a `torch$utils$data$Dataset` object,
#' or a list of arrays with the same length.
#'
#' @param left_size
#' If float (in the range `[0, 1]`), it signifies
#' the fraction of the data to pack in the left dataset. If integer, it
#' signifies the number of samples to pack in the left dataset. If
#' `NULL`, defaults to the complement to `right_size`.
#' Defaults to `NULL`.
#'
#' @param right_size
#' If float (in the range `[0, 1]`), it signifies
#' the fraction of the data to pack in the right dataset.
#' If integer, it signifies the number of samples to pack
#' in the right dataset.
#' If `NULL`, defaults to the complement to `left_size`.
#' Defaults to `NULL`.
#'
#' @param shuffle
#' Boolean, whether to shuffle the data before splitting it.
#'
#' @param seed
#' A random seed for shuffling.
#'
#' @export
#' @family dataset utils
#' @family utils
#' @seealso
#' + <https://keras.io/api/utils/python_utils#splitdataset-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/split_dataset>
#'
#' @tether keras.utils.split_dataset
split_dataset <-
function (dataset, left_size = NULL, right_size = NULL, shuffle = FALSE,
          seed = NULL)
{
  args <- capture_args(list(left_size = function (x)
    ifelse(x < 1, x, as_integer(x)), right_size = function (x)
      ifelse(x < 1, x, as_integer(x)), seed = as_integer))
  do.call(keras$utils$split_dataset, args)
}



#' Generates a `tf.data.Dataset` from image files in a directory.
#'
#' @description
#' If your directory structure is:
#'
#' ```
#' main_directory/
#' ...class_a/
#' ......a_image_1.jpg
#' ......a_image_2.jpg
#' ...class_b/
#' ......b_image_1.jpg
#' ......b_image_2.jpg
#' ```
#'
#' Then calling `image_dataset_from_directory(main_directory,
#' labels = 'inferred')` will return a `tf.data.Dataset` that yields batches of
#' images from the subdirectories `class_a` and `class_b`, together with labels
#' 0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).
#'
#' Supported image formats: `.jpeg`, `.jpg`, `.png`, `.bmp`, `.gif`.
#' Animated gifs are truncated to the first frame.
#'
#' @returns
#' A `tf.data.Dataset` object.
#'
#' - If `label_mode` is `NULL`, it yields `float32` tensors of shape
#'     `(batch_size, image_size[1], image_size[2], num_channels)`,
#'     encoding images (see below for rules regarding `num_channels`).
#' - Otherwise, it yields a tuple `(images, labels)`, where `images` has
#'     shape `(batch_size, image_size[1], image_size[2], num_channels)`,
#'     and `labels` follows the format described below.
#'
#' Rules regarding labels format:
#'
#' - if `label_mode` is `"int"`, the labels are an `int32` tensor of shape
#'     `(batch_size,)`.
#' - if `label_mode` is `"binary"`, the labels are a `float32` tensor of
#'     1s and 0s of shape `(batch_size, 1)`.
#' - if `label_mode` is `"categorical"`, the labels are a `float32` tensor
#'     of shape `(batch_size, num_classes)`, representing a one-hot
#'     encoding of the class index.
#'
#' Rules regarding number of channels in the yielded images:
#'
#' - if `color_mode` is `"grayscale"`,
#'     there's 1 channel in the image tensors.
#' - if `color_mode` is `"rgb"`,
#'     there are 3 channels in the image tensors.
#' - if `color_mode` is `"rgba"`,
#'     there are 4 channels in the image tensors.
#'
#' @param directory
#' Directory where the data is located.
#' If `labels` is `"inferred"`, it should contain
#' subdirectories, each containing images for a class.
#' Otherwise, the directory structure is ignored.
#'
#' @param labels
#' Either `"inferred"`
#' (labels are generated from the directory structure),
#' `NULL` (no labels),
#' or a list/tuple of integer labels of the same size as the number of
#' image files found in the directory. Labels should be sorted
#' according to the alphanumeric order of the image file paths
#' (obtained via `os.walk(directory)` in Python).
#'
#' @param label_mode
#' String describing the encoding of `labels`. Options are:
#' - `"int"`: means that the labels are encoded as integers
#'     (e.g. for `sparse_categorical_crossentropy` loss).
#' - `"categorical"` means that the labels are
#'     encoded as a categorical vector
#'     (e.g. for `categorical_crossentropy` loss).
#' - `"binary"` means that the labels (there can be only 2)
#'     are encoded as `float32` scalars with values 0 or 1
#'     (e.g. for `binary_crossentropy`).
#' - `NULL` (no labels).
#'
#' @param class_names
#' Only valid if `labels` is `"inferred"`.
#' This is the explicit list of class names
#' (must match names of subdirectories). Used to control the order
#' of the classes (otherwise alphanumerical order is used).
#'
#' @param color_mode
#' One of `"grayscale"`, `"rgb"`, `"rgba"`.
#' Whether the images will be converted to
#' have 1, 3, or 4 channels. Defaults to `"rgb"`.
#'
#' @param batch_size
#' Size of the batches of data. Defaults to 32.
#' If `NULL`, the data will not be batched
#' (the dataset will yield individual samples).
#'
#' @param image_size
#' Size to resize images to after they are read from disk,
#' specified as `(height, width)`.
#' Since the pipeline processes batches of images that must all have
#' the same size, this must be provided. Defaults to `(256, 256)`.
#'
#' @param shuffle
#' Whether to shuffle the data. Defaults to `TRUE`.
#' If set to `FALSE`, sorts the data in alphanumeric order.
#'
#' @param seed
#' Optional random seed for shuffling and transformations.
#'
#' @param validation_split
#' Optional float between 0 and 1,
#' fraction of data to reserve for validation.
#'
#' @param subset
#' Subset of the data to return.
#' One of `"training"`, `"validation"`, or `"both"`.
#' Only used if `validation_split` is set.
#' When `subset = "both"`, the utility returns a tuple of two datasets
#' (the training and validation datasets respectively).
#'
#' @param interpolation
#' String, the interpolation method used when
#' resizing images.
#' Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,
#' `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
#' Defaults to `"bilinear"`.
#'
#' @param follow_links
#' Whether to visit subdirectories pointed to by symlinks.
#' Defaults to `FALSE`.
#'
#' @param crop_to_aspect_ratio
#' If `TRUE`, resize the images without aspect
#' ratio distortion. When the original aspect ratio differs from the
#' target aspect ratio, the output image will be cropped so as to
#' return the largest possible window in the image
#' (of size `image_size`) that matches the target aspect ratio. By
#' default (`crop_to_aspect_ratio = FALSE`), aspect ratio may not be
#' preserved.
#'
#' @param pad_to_aspect_ratio
#' If `TRUE`, resize the images without aspect
#' ratio distortion. When the original aspect ratio differs from the
#' target aspect ratio, the output image will be padded so as to
#' return the largest possible window in the image
#' (of size `image_size`) that matches the target aspect ratio. By
#' default (`pad_to_aspect_ratio=FALSE`), aspect ratio may not be
#' preserved.
#'
#' @param data_format
#' If `NULL` uses [`config_image_data_format()`]
#' otherwise either `'channel_last'` or `'channel_first'`.
#'
#' @param verbose
#' Whether to display number information on classes and
#' number of files found. Defaults to `TRUE`.
#'
#' @export
#' @family dataset utils
#' @family image dataset utils
#' @family utils
#' @family preprocessing
#' @seealso
#' + <https://keras.io/api/data_loading/image#imagedatasetfromdirectory-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory>
#' @tether keras.utils.image_dataset_from_directory
image_dataset_from_directory <-
function (directory, labels = "inferred", label_mode = "int",
          class_names = NULL, color_mode = "rgb", batch_size = 32L,
          image_size = c(256L, 256L), shuffle = TRUE, seed = NULL,
          validation_split = NULL, subset = NULL, interpolation = "bilinear",
          follow_links = FALSE, crop_to_aspect_ratio = FALSE,
          pad_to_aspect_ratio = FALSE, data_format = NULL, verbose = TRUE)
{
  args <- capture_args(list(labels = as_integer,
                            image_size = function(x) lapply(x, as_integer),
                            batch_size = as_integer, seed = as_integer))
  do.call(keras$utils$image_dataset_from_directory, args)
}



#' Generates a `tf.data.Dataset` from text files in a directory.
#'
#' @description
#' If your directory structure is:
#'
#' ```
#' main_directory/
#' ...class_a/
#' ......a_text_1.txt
#' ......a_text_2.txt
#' ...class_b/
#' ......b_text_1.txt
#' ......b_text_2.txt
#' ```
#'
#' Then calling `text_dataset_from_directory(main_directory,
#' labels='inferred')` will return a `tf.data.Dataset` that yields batches of
#' texts from the subdirectories `class_a` and `class_b`, together with labels
#' 0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).
#'
#' Only `.txt` files are supported at this time.
#'
#' @returns
#' A `tf.data.Dataset` object.
#'
#' - If `label_mode` is `NULL`, it yields `string` tensors of shape
#'     `(batch_size,)`, containing the contents of a batch of text files.
#' - Otherwise, it yields a tuple `(texts, labels)`, where `texts`
#'     has shape `(batch_size,)` and `labels` follows the format described
#'     below.
#'
#' Rules regarding labels format:
#'
#' - if `label_mode` is `int`, the labels are an `int32` tensor of shape
#'     `(batch_size,)`.
#' - if `label_mode` is `binary`, the labels are a `float32` tensor of
#'     1s and 0s of shape `(batch_size, 1)`.
#' - if `label_mode` is `categorical`, the labels are a `float32` tensor
#'     of shape `(batch_size, num_classes)`, representing a one-hot
#'     encoding of the class index.
#'
#' @param directory
#' Directory where the data is located.
#' If `labels` is `"inferred"`, it should contain
#' subdirectories, each containing text files for a class.
#' Otherwise, the directory structure is ignored.
#'
#' @param labels
#' Either `"inferred"`
#' (labels are generated from the directory structure),
#' `NULL` (no labels),
#' or a list/tuple of integer labels of the same size as the number of
#' text files found in the directory. Labels should be sorted according
#' to the alphanumeric order of the text file paths
#' (obtained via `os.walk(directory)` in Python).
#'
#' @param label_mode
#' String describing the encoding of `labels`. Options are:
#' - `"int"`: means that the labels are encoded as integers
#'     (e.g. for `sparse_categorical_crossentropy` loss).
#' - `"categorical"` means that the labels are
#'     encoded as a categorical vector
#'     (e.g. for `categorical_crossentropy` loss).
#' - `"binary"` means that the labels (there can be only 2)
#'     are encoded as `float32` scalars with values 0 or 1
#'     (e.g. for `binary_crossentropy`).
#' - `NULL` (no labels).
#'
#' @param class_names
#' Only valid if `"labels"` is `"inferred"`.
#' This is the explicit list of class names
#' (must match names of subdirectories). Used to control the order
#' of the classes (otherwise alphanumerical order is used).
#'
#' @param batch_size
#' Size of the batches of data.
#' If `NULL`, the data will not be batched
#' (the dataset will yield individual samples).
#' Defaults to `32`.
#'
#' @param max_length
#' Maximum size of a text string. Texts longer than this will
#' be truncated to `max_length`.
#'
#' @param shuffle
#' Whether to shuffle the data.
#' If set to `FALSE`, sorts the data in alphanumeric order.
#' Defaults to `TRUE`.
#'
#' @param seed
#' Optional random seed for shuffling and transformations.
#'
#' @param validation_split
#' Optional float between 0 and 1,
#' fraction of data to reserve for validation.
#'
#' @param subset
#' Subset of the data to return.
#' One of `"training"`, `"validation"` or `"both"`.
#' Only used if `validation_split` is set.
#' When `subset="both"`, the utility returns a tuple of two datasets
#' (the training and validation datasets respectively).
#'
#' @param follow_links
#' Whether to visits subdirectories pointed to by symlinks.
#' Defaults to `FALSE`.
#'
#' @param verbose
#' Whether to display number information on classes and
#' number of files found. Defaults to `TRUE`.
#'
#' @export
#' @family dataset utils
#' @family text dataset utils
#' @family utils
#' @family preprocessing
#' @seealso
#' + <https://keras.io/api/data_loading/text#textdatasetfromdirectory-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/text_dataset_from_directory>
#' @tether keras.utils.text_dataset_from_directory
text_dataset_from_directory <-
function (directory, labels = "inferred", label_mode = "int",
    class_names = NULL, batch_size = 32L, max_length = NULL,
    shuffle = TRUE, seed = NULL, validation_split = NULL, subset = NULL,
    follow_links = FALSE, verbose = TRUE)
{
    args <- capture_args(list(labels = as_integer, label_mode = as_integer,
        batch_size = as_integer, seed = as_integer))
    do.call(keras$utils$text_dataset_from_directory, args)
}


#' Creates a dataset of sliding windows over a timeseries provided as array.
#'
#' @description
#' This function takes in a sequence of data-points gathered at
#' equal intervals, along with time series parameters such as
#' length of the sequences/windows, spacing between two sequence/windows, etc.,
#' to produce batches of timeseries inputs and targets.
#'
#' @returns
#' A `tf$data$Dataset` instance. If `targets` was passed, the dataset yields
#' list `(batch_of_sequences, batch_of_targets)`. If not, the dataset yields
#' only `batch_of_sequences`.
#'
#' Example 1:
#'
#' Consider indices `[0, 1, ... 98]`.
#' With `sequence_length=10,  sampling_rate=2, sequence_stride=3`,
#' `shuffle=FALSE`, the dataset will yield batches of sequences
#' composed of the following indices:
#'
#' ```
#' First sequence:  [0  2  4  6  8 10 12 14 16 18]
#' Second sequence: [3  5  7  9 11 13 15 17 19 21]
#' Third sequence:  [6  8 10 12 14 16 18 20 22 24]
#' ...
#' Last sequence:   [78 80 82 84 86 88 90 92 94 96]
#' ```
#'
#' In this case the last 2 data points are discarded since no full sequence
#' can be generated to include them (the next sequence would have started
#' at index 81, and thus its last step would have gone over 98).
#'
#' Example 2: Temporal regression.
#'
#' Consider an array `data` of scalar values, of shape `(steps,)`.
#' To generate a dataset that uses the past 10
#' timesteps to predict the next timestep, you would use:
#'
#' ```{r}
#' data <- op_array(1:20)
#' input_data <- data[1:10]
#' targets <- data[11:20]
#' dataset <- timeseries_dataset_from_array(
#'   input_data, targets, sequence_length=10)
#' iter <- reticulate::as_iterator(dataset)
#' reticulate::iter_next(iter)
#' ```
#'
#' Example 3: Temporal regression for many-to-many architectures.
#'
#' Consider two arrays of scalar values `X` and `Y`,
#' both of shape `(100,)`. The resulting dataset should consist samples with
#' 20 timestamps each. The samples should not overlap.
#' To generate a dataset that uses the current timestamp
#' to predict the corresponding target timestep, you would use:
#'
#' ```{r}
#' X <- op_array(1:100)
#' Y <- X*2
#'
#' sample_length <- 20
#' input_dataset <- timeseries_dataset_from_array(
#'     X, NULL, sequence_length=sample_length, sequence_stride=sample_length)
#' target_dataset <- timeseries_dataset_from_array(
#'     Y, NULL, sequence_length=sample_length, sequence_stride=sample_length)
#'
#'
#' inputs <- reticulate::as_iterator(input_dataset) %>% reticulate::iter_next()
#' targets <- reticulate::as_iterator(target_dataset) %>% reticulate::iter_next()
#' ```
#'
#' @param data
#' array or eager tensor
#' containing consecutive data points (timesteps).
#' The first dimension is expected to be the time dimension.
#'
#' @param targets
#' Targets corresponding to timesteps in `data`.
#' `targets[i]` should be the target
#' corresponding to the window that starts at index `i`
#' (see example 2 below).
#' Pass `NULL` if you don't have target data (in this case the dataset
#' will only yield the input data).
#'
#' @param sequence_length
#' Length of the output sequences
#' (in number of timesteps).
#'
#' @param sequence_stride
#' Period between successive output sequences.
#' For stride `s`, output samples would
#' start at index `data[i]`, `data[i + s]`, `data[i + 2 * s]`, etc.
#'
#' @param sampling_rate
#' Period between successive individual timesteps
#' within sequences. For rate `r`, timesteps
#' `data[i], data[i + r], ... data[i + sequence_length]`
#' are used for creating a sample sequence.
#'
#' @param batch_size
#' Number of timeseries samples in each batch
#' (except maybe the last one). If `NULL`, the data will not be batched
#' (the dataset will yield individual samples).
#'
#' @param shuffle
#' Whether to shuffle output samples,
#' or instead draw them in chronological order.
#'
#' @param seed
#' Optional int; random seed for shuffling.
#'
#' @param start_index
#' Optional int; data points earlier (exclusive)
#' than `start_index` will not be used
#' in the output sequences. This is useful to reserve part of the
#' data for test or validation.
#'
#' @param end_index
#' Optional int; data points later (exclusive) than `end_index`
#' will not be used in the output sequences.
#' This is useful to reserve part of the data for test or validation.
#'
#' @export
#' @family dataset utils
#' @family timesery dataset utils
#' @family utils
#' @family preprocessing
#' @seealso
#' + <https://keras.io/api/data_loading/timeseries#timeseriesdatasetfromarray-function>
#  + <https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array>
#'
#' @tether keras.utils.timeseries_dataset_from_array
timeseries_dataset_from_array <-
function (data, targets, sequence_length, sequence_stride = 1L,
    sampling_rate = 1L, batch_size = 128L, shuffle = FALSE, seed = NULL,
    start_index = NULL, end_index = NULL)
{
    args <- capture_args(list(sequence_stride = as_integer,
        sampling_rate = as_integer, batch_size = as_integer,
        seed = as_integer, start_index = as_integer, end_index = as_integer,
        data = keras_array, targets = keras_array))
    do.call(keras$utils$timeseries_dataset_from_array, args)
}
