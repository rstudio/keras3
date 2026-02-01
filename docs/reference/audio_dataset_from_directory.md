# Generates a `tf.data.Dataset` from audio files in a directory.

If your directory structure is:

    main_directory/
    ...class_a/
    ......a_audio_1.wav
    ......a_audio_2.wav
    ...class_b/
    ......b_audio_1.wav
    ......b_audio_2.wav

Then calling
`audio_dataset_from_directory(main_directory, labels = 'inferred')` will
return a `tf.data.Dataset` that yields batches of audio files from the
subdirectories `class_a` and `class_b`, together with labels 0 and 1 (0
corresponding to `class_a` and 1 corresponding to `class_b`).

Only `.wav` files are supported at this time.

## Usage

``` r
audio_dataset_from_directory(
  directory,
  labels = "inferred",
  label_mode = "int",
  class_names = NULL,
  batch_size = 32L,
  sampling_rate = NULL,
  output_sequence_length = NULL,
  ragged = FALSE,
  shuffle = TRUE,
  seed = NULL,
  validation_split = NULL,
  subset = NULL,
  follow_links = FALSE,
  verbose = TRUE
)
```

## Arguments

- directory:

  Directory where the data is located. If `labels` is `"inferred"`, it
  should contain subdirectories, each containing audio files for a
  class. Otherwise, the directory structure is ignored.

- labels:

  Either "inferred" (labels are generated from the directory structure),
  `NULL` (no labels), or a list/tuple of integer labels of the same size
  as the number of audio files found in the directory. Labels should be
  sorted according to the alphanumeric order of the audio file paths
  (obtained via `os.walk(directory)` in Python).

- label_mode:

  String describing the encoding of `labels`. Options are:

  - `"int"`: means that the labels are encoded as integers (e.g. for
    `sparse_categorical_crossentropy` loss).

  - `"categorical"` means that the labels are encoded as a categorical
    vector (e.g. for `categorical_crossentropy` loss)

  - `"binary"` means that the labels (there can be only 2) are encoded
    as `float32` scalars with values 0 or 1 (e.g. for
    `binary_crossentropy`).

  - `NULL` (no labels).

- class_names:

  Only valid if "labels" is `"inferred"`. This is the explicit list of
  class names (must match names of subdirectories). Used to control the
  order of the classes (otherwise alphanumerical order is used).

- batch_size:

  Size of the batches of data. Default: 32. If `NULL`, the data will not
  be batched (the dataset will yield individual samples).

- sampling_rate:

  Audio sampling rate (in samples per second).

- output_sequence_length:

  Maximum length of an audio sequence. Audio files longer than this will
  be truncated to `output_sequence_length`. If set to `NULL`, then all
  sequences in the same batch will be padded to the length of the
  longest sequence in the batch.

- ragged:

  Whether to return a Ragged dataset (where each sequence has its own
  length). Defaults to `FALSE`.

- shuffle:

  Whether to shuffle the data. Defaults to `TRUE`. If set to `FALSE`,
  sorts the data in alphanumeric order.

- seed:

  Optional random seed for shuffling and transformations.

- validation_split:

  Optional float between 0 and 1, fraction of data to reserve for
  validation.

- subset:

  Subset of the data to return. One of `"training"`, `"validation"` or
  `"both"`. Only used if `validation_split` is set.

- follow_links:

  Whether to visits subdirectories pointed to by symlinks. Defaults to
  `FALSE`.

- verbose:

  Whether to display number information on classes and number of files
  found. Defaults to `TRUE`.

## Value

A `tf.data.Dataset` object.

- If `label_mode` is `NULL`, it yields `string` tensors of shape
  `(batch_size,)`, containing the contents of a batch of audio files.

- Otherwise, it yields a tuple `(audio, labels)`, where `audio` has
  shape `(batch_size, sequence_length, num_channels)` and `labels`
  follows the format described below.

Rules regarding labels format:

- if `label_mode` is `int`, the labels are an `int32` tensor of shape
  `(batch_size,)`.

- if `label_mode` is `binary`, the labels are a `float32` tensor of 1s
  and 0s of shape `(batch_size, 1)`.

- if `label_mode` is `categorical`, the labels are a `float32` tensor of
  shape `(batch_size, num_classes)`, representing a one-hot encoding of
  the class index.

## See also

- <https://keras.io/api/data_loading/audio#audiodatasetfromdirectory-function>

Other dataset utils:  
[`image_dataset_from_directory()`](https://keras3.posit.co/reference/image_dataset_from_directory.md)  
[`split_dataset()`](https://keras3.posit.co/reference/split_dataset.md)  
[`text_dataset_from_directory()`](https://keras3.posit.co/reference/text_dataset_from_directory.md)  
[`timeseries_dataset_from_array()`](https://keras3.posit.co/reference/timeseries_dataset_from_array.md)  

Other utils:  
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
[`layer_feature_space()`](https://keras3.posit.co/reference/layer_feature_space.md)  
[`normalize()`](https://keras3.posit.co/reference/normalize.md)  
[`pad_sequences()`](https://keras3.posit.co/reference/pad_sequences.md)  
[`set_random_seed()`](https://keras3.posit.co/reference/set_random_seed.md)  
[`split_dataset()`](https://keras3.posit.co/reference/split_dataset.md)  
[`text_dataset_from_directory()`](https://keras3.posit.co/reference/text_dataset_from_directory.md)  
[`timeseries_dataset_from_array()`](https://keras3.posit.co/reference/timeseries_dataset_from_array.md)  
[`to_categorical()`](https://keras3.posit.co/reference/to_categorical.md)  
[`zip_lists()`](https://keras3.posit.co/reference/zip_lists.md)  
