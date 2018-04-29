
## Keras 2.1.6 (CRAN)

- Fix issue with single-element vectors passed to text preprocessing functions

- Compatibility with TensorFlow v1.7 Keras implementation

- Support `workers` parameter for native Keras generators (e.g. `flow_images_from_directory()`)

- Accept tensor as argument to `k_pow()` 

- In `callback_reduce_lr_on_plateau()`, rename `epsilon` argument to `min_delta` 
  (backwards-compatible).

- Add `axis` parameter to `k_softmax()`

- Add `send_as_json` parameter to `callback_remote_monitor()`

- Add `data_format` method to `layer_flatten()`

- In `multi_gpu_model()`, add arguments `cpu_merge` and `cpu_relocation` (controlling whether
  to force the template model's weights to be on CPU, and whether to operate merge operations
  on CPU or GPU).
  
- Record correct loss name for tfruns when custom functions are provided for `loss`


## Keras 2.1.5

- Support for custom constraints from R 

- Added `timeseries_generator()` utility function 

- New layer `layer_depthwise_conv_2d()`

- Added `brightness_range` and `validation_split` arguments to 
  [image_data_generator()].


## Keras 2.1.4

- Added support for `remove_learning_phase` in `export_savedmodel()` to avoid
  removing learning phase.
  
- Normalize validation data to Keras array in `fit()` and `fit_generator()`

- Ensure that custom layers return a tuple from `compute_output_shape()`

- Added Nasnet and Densenet pre-trained models

- New layers `layer_activation_softmax()` and `layer_separable_conv_1d()`

- Added `amsgrad` parameter to `optimizer_adam()` 

- Fix incompatibility with Progbar.update() method in Keras 2.1.4


## Keras 2.1.3

- Models saved via `export_savedmodel()` that make use of learning phases can
  now be exported without having to manually reload the original model.

- Ensure that models saved via `export_savedmodel()` can be served from CloudML

- Run image data generators with R preprocessing functions on the main thread

- Return R list from `texts_to_sequences()`

- Various fixes for `use_implementation()` function


## Keras 2.1.2 

- Added `theme_bw` option to plot method for training history 

- Support TF Dataset objects as generators for `fit_generator()`, etc.

- Added `use_implementation()` and `use_backend()` functions as alternative to
  setting `KERAS_IMPLEMENATION` and `KERAS_BACKEND` environment variables.

- Added R wrappers for Keras backend functions (e.g. `k_variable()`, 
  `k_dot()`, etc.)

- Use 1-based axis for `normalize` function.

- Fix issue with printing training history after early stopping.

- Experimental support for using the PlaidML backend.

- Correct handling for R functions specified in `custom_objects`

- Added `with_custom_object_scope()` function.

- Automatically provide name to loss function during compile
  (enables save/load of models with custom loss function)
  
- Provide global `keras.fit_verbose` option (defaults to 1)


## keras 2.0.9

- Added `multi_gpu_model()` function.

- Automatically call `keras_array()` on the results of generator functions.

- Ensure that `steps_per_epoch` is passed as an integer

- Import `evaluate()` generic from tensorflow package

- Handle `NULL` when converting R arrays to Keras friendly arrays

- Added `dataset_imbd_word_index()` function

- Ensure that `sample_weight` is passed to `fit()` as an array.

- Accept single function as `metrics` argument to `compile()`

- Automatically cast `input_shape` argument to applications to integer

- Allow Keras models to be composable within model pipelines

- Added `freeze_weights()` and `unfreeze_weights()` functions.

- Implement `export_savedmodel()` generic from TensorFlow package

- Convert R arrays to row-major before image preprocessing

- Use `tensorflow.keras` for tensorflow implementation (TF v1.4)

- Added `application_inception_resnet_v2()` pre-trained model

- Added `dataset_fashion_mnist()` dataset

- Added `layer_cudnn_gru()` and `layer_cudnn_lstm()` (faster 
  recurrent layers backed by [CuDNN](https://developer.nvidia.com/cudnn))
  
- Added `layer_minimum()` function

- Added `interpolation` parameter to `image_load()` function

- Add `save_text_tokenizer()` and `load_text_tokenizer()` functions.

- Fix for progress bar output in Keras >= 2.0.9

- Remove deprecated `implementation` argument from recurrent layers

- Support for passing generators for validation data in `fit_generator()`

- Accept single integer arguments for kernel sizes

- Add standard layer arguments to `layer_flatten()` and `layer_separable_conv_2d()`

- Added `image_array_resize()` and `image_array_save()` for 3D image arrays.

- Allow custom layers and lambda layers to accept list parameters.

- Expose `add_loss()` function for custom layers


## keras 2.0.8

- Add `use_session_with_seed()` function that establishes a random seed for the Keras session.
  Note that this should not be used when training time is paramount, as it disables GPU
  computation and CPU parallelism by default for more deterministic computations.

- Fix for plotting training history with early stopping callback (thanks to @JamesAllingham).

- Return R training history object from `fit_generator()`

- Rename `to_numpy_array()` function to `keras_array()` reflecting automatic use
  of Keras default backend float type and "C" ordering.
  
- Add standard layer arguments (e.g. `name`, `trainable`, etc.) to merge layers

- Better support for training models from data tensors in TensorFlow (e.g. Datasets, TFRecords). Add a related example script.

- Add `clone_model()` function, enabling to construct a new model, given an existing model to use as a template. Works even in a TensorFlow graph different from that of the original model.

- Add `target_tensors` argument in `compile()`, enabling to use custom tensors or placeholders as model targets.

- Add `steps_per_epoch` argument in `fit()`, enabling to train a model from data tensors in a way that is consistent with training from arrays. Similarly, add `steps` argument in `predict()` and `evaluate()`.

- Add `layer_subtract()` layer function.

- Add `weighted_metrics` argument in compile to specify metric functions meant to take into account `sample_weight` or `class_weight`.

- Enable stateful RNNs with CNTK.


## keras 2.0.6

- `install_keras()` function which installs both TensorFlow and Keras

- Use keras package as default implementation rather than tf.contrib.keras

- Training metrics plotted in realtime within the RStudio Viewer during fit 

- `serialize_model()` and `unserialize_model()` functions for saving 
  Keras models as 'raw' R objects.

- Automatically convert 64-bit R floats to backend default float type

- Ensure that arrays passed to generator functions are normalized to C-order 

- `to_numpy_array()` utility function for custom generators (enables
  custom generators to yield C-ordered arrays of the correct float type)
  
- Added `batch_size` and `write_grads` arguments to `callback_tensorboard()`

- Added `return_state` argument to recurrent layers.

- Don't re-export `install_tensorflow()` and `tf_config()` from tensorflow
  package.

- `is_keras_available()` function to probe whether the Keras python 
  package is available in the current environment.
  
- `as.data.frame()` S3 method for Keras training history

- Remove names from `keras_model()` inputs

- Return result of `evaluate()` as named list

- Write run metrics and evaluation data to tfruns

- Provide hint to use r-tensorflow environment when importing keras


## keras 2.0.5

- Initial CRAN release

