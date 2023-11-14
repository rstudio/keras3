Generates a `tf.data.Dataset` from audio files in a directory.

@description
If your directory structure is:

```
main_directory/
...class_a/
......a_audio_1.wav
......a_audio_2.wav
...class_b/
......b_audio_1.wav
......b_audio_2.wav
```

Then calling `audio_dataset_from_directory(main_directory,
labels='inferred')`
will return a `tf.data.Dataset` that yields batches of audio files from
the subdirectories `class_a` and `class_b`, together with labels
0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

Only `.wav` files are supported at this time.

@returns
A `tf.data.Dataset` object.

- If `label_mode` is `None`, it yields `string` tensors of shape
  `(batch_size,)`, containing the contents of a batch of audio files.
- Otherwise, it yields a tuple `(audio, labels)`, where `audio`
  has shape `(batch_size, sequence_length, num_channels)` and `labels`
  follows the format described
  below.

Rules regarding labels format:

- if `label_mode` is `int`, the labels are an `int32` tensor of shape
  `(batch_size,)`.
- if `label_mode` is `binary`, the labels are a `float32` tensor of
  1s and 0s of shape `(batch_size, 1)`.
- if `label_mode` is `categorical`, the labels are a `float32` tensor
  of shape `(batch_size, num_classes)`, representing a one-hot
  encoding of the class index.

@param directory
Directory where the data is located.
If `labels` is `"inferred"`, it should contain subdirectories,
each containing audio files for a class. Otherwise, the directory
structure is ignored.

@param labels
Either "inferred" (labels are generated from the directory
structure), `None` (no labels), or a list/tuple of integer labels
of the same size as the number of audio files found in
the directory. Labels should be sorted according to the
alphanumeric order of the audio file paths
(obtained via `os.walk(directory)` in Python).

@param label_mode
String describing the encoding of `labels`. Options are:
- `"int"`: means that the labels are encoded as integers (e.g. for
  `sparse_categorical_crossentropy` loss).
- `"categorical"` means that the labels are encoded as a categorical
  vector (e.g. for `categorical_crossentropy` loss)
- `"binary"` means that the labels (there can be only 2)
  are encoded as `float32` scalars with values 0
  or 1 (e.g. for `binary_crossentropy`).
- `None` (no labels).

@param class_names
Only valid if "labels" is `"inferred"`.
This is the explicit list of class names
(must match names of subdirectories). Used to control the order
of the classes (otherwise alphanumerical order is used).

@param batch_size
Size of the batches of data. Default: 32. If `None`,
the data will not be batched
(the dataset will yield individual samples).

@param sampling_rate
Audio sampling rate (in samples per second).

@param output_sequence_length
Maximum length of an audio sequence. Audio files
longer than this will be truncated to `output_sequence_length`.
If set to `None`, then all sequences in the same batch will
be padded to the
length of the longest sequence in the batch.

@param ragged
Whether to return a Ragged dataset (where each sequence has its
own length). Defaults to `False`.

@param shuffle
Whether to shuffle the data. Defaults to `True`.
If set to `False`, sorts the data in alphanumeric order.

@param seed
Optional random seed for shuffling and transformations.

@param validation_split
Optional float between 0 and 1, fraction of data to
reserve for validation.

@param subset
Subset of the data to return. One of `"training"`,
`"validation"` or `"both"`. Only used if `validation_split` is set.

@param follow_links
Whether to visits subdirectories pointed to by symlinks.
Defaults to `False`.

@export
@family dataset audio utils
@family audio utils
@family utils
@seealso
+ <https:/keras.io/api/data_loading/audio#audiodatasetfromdirectory-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/audio_dataset_from_directory>
