# Sets all random seeds (Python, NumPy, and backend framework, e.g. TF).

You can use this utility to make almost any Keras program fully
deterministic. Some limitations apply in cases where network
communications are involved (e.g. parameter server distribution), which
creates additional sources of randomness, or when certain
non-deterministic cuDNN ops are involved.

This sets:

- the R session seed: [`set.seed()`](https://rdrr.io/r/base/Random.html)

- the Python session seed: `import random; random.seed(seed)`

- the Python NumPy seed: `import numpy; numpy.random.seed(seed)`

- the TensorFlow seed: `tf$random$set_seed(seed)` (only if TF is
  installed)

- The Torch seed: `import("torch")$manual_seed(seed)` (only if the
  backend is torch)

- and disables Python hash randomization.

Note that the TensorFlow seed is set even if you're not using TensorFlow
as your backend framework, since many workflows leverage `tf$data`
pipelines (which feature random shuffling). Likewise many workflows
might leverage NumPy APIs.

## Usage

``` r
set_random_seed(seed)
```

## Arguments

- seed:

  Integer, the random seed to use.

## Value

No return value, called for side effects.

## See also

- <https://keras.io/api/utils/python_utils#setrandomseed-function>

Other utils:  
[`audio_dataset_from_directory()`](https://keras3.posit.co/dev/reference/audio_dataset_from_directory.md)  
[`clear_session()`](https://keras3.posit.co/dev/reference/clear_session.md)  
[`config_disable_interactive_logging()`](https://keras3.posit.co/dev/reference/config_disable_interactive_logging.md)  
[`config_disable_traceback_filtering()`](https://keras3.posit.co/dev/reference/config_disable_traceback_filtering.md)  
[`config_enable_interactive_logging()`](https://keras3.posit.co/dev/reference/config_enable_interactive_logging.md)  
[`config_enable_traceback_filtering()`](https://keras3.posit.co/dev/reference/config_enable_traceback_filtering.md)  
[`config_is_interactive_logging_enabled()`](https://keras3.posit.co/dev/reference/config_is_interactive_logging_enabled.md)  
[`config_is_traceback_filtering_enabled()`](https://keras3.posit.co/dev/reference/config_is_traceback_filtering_enabled.md)  
[`get_file()`](https://keras3.posit.co/dev/reference/get_file.md)  
[`get_source_inputs()`](https://keras3.posit.co/dev/reference/get_source_inputs.md)  
[`image_array_save()`](https://keras3.posit.co/dev/reference/image_array_save.md)  
[`image_dataset_from_directory()`](https://keras3.posit.co/dev/reference/image_dataset_from_directory.md)  
[`image_from_array()`](https://keras3.posit.co/dev/reference/image_from_array.md)  
[`image_load()`](https://keras3.posit.co/dev/reference/image_load.md)  
[`image_smart_resize()`](https://keras3.posit.co/dev/reference/image_smart_resize.md)  
[`image_to_array()`](https://keras3.posit.co/dev/reference/image_to_array.md)  
[`layer_feature_space()`](https://keras3.posit.co/dev/reference/layer_feature_space.md)  
[`normalize()`](https://keras3.posit.co/dev/reference/normalize.md)  
[`pad_sequences()`](https://keras3.posit.co/dev/reference/pad_sequences.md)  
[`split_dataset()`](https://keras3.posit.co/dev/reference/split_dataset.md)  
[`text_dataset_from_directory()`](https://keras3.posit.co/dev/reference/text_dataset_from_directory.md)  
[`timeseries_dataset_from_array()`](https://keras3.posit.co/dev/reference/timeseries_dataset_from_array.md)  
[`to_categorical()`](https://keras3.posit.co/dev/reference/to_categorical.md)  
[`zip_lists()`](https://keras3.posit.co/dev/reference/zip_lists.md)  
