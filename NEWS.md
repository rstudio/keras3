# keras 2.16.0

- Updates for usage with Legacy Keras (#1515). 

  - A package startup message is now displayed encouraging users to migrate to keras3.
  - New function `py_require_legacy_keras()`.
  - `install_keras()` now installs legacy keras `tf-keras`.
  - `TF_USE_LEGACY_KERAS=1` envvar is now set on package startup.

- Documentation updates for CRAN (#1514)

# keras 2.15.0

- Default TensorFlow/Keras version installed by `install_keras()` is now 2.15. 
  This is the last Tensorflow version where where Keras 2 is the default. 
  To use Keras with Tensorflow v2.16 and up, use the new {keras3} R package.

- Updates to allow both R packages {keras} and {keras3} to be loaded.

- Updates for R-devel (4.4).


# keras 2.13.0

- Default TF version installed by `install_keras()` is now 2.13.

- Updated layers:
  - `layer_batch_normalization()` updated signature, with changes to options for distributed training.
  - `layer_embedding()` gains a `sparse` argument.

- Fixed deadlock when an R generator was passed to `fit()`, `predict()`, and other endpoints.

- When `fit(verbose = "auto")` is evaluated in the context of a knitr document 
  (e.g., quarto or rmarkdown document being rendered), verbose will now 
  default to `2`, showing one line per epoch.

# keras 2.11.1

- Update S3 method formals per new CRAN requirement (`r_to_py.keras_layer_wrapper()`)

- Fixed an issue where `get_file()` would place incorrectly 
  save files in the current working directory. (#1365)

# keras 2.11.0

- Default TensorFlow version installed by `install_keras()` is now 2.11. 

- All optimizers have been updated for keras/tensorflow version 2.11.
  Arguments to all the optimizers have changed. To access the previous
  optimizer implementations, use the constructors available at
  `keras$optimizers$legacy`. For example, use `keras$optimizers$legacy$Adam()`
  for the previous implementation of `optimizer_adam()`.

- New optimizer `optimizer_frtl()`.

- updates to layers:
  - `layer_attention()` gains `score_mode` and `dropout` arguments.
  - `layer_discretization()` gains `output_mode` and `sparse` arguments.
  - `layer_gaussian_dropout()` and `layer_gaussian_noise()` gain a `seed` argument.
  - `layer_hashing()` gains `output_mode` and `sparse` arguments.
  - `layer_integer_lookup()` gains `vocabulary_dtype` and `idf_weights` arguments.
  - `layer_normalization()` gains an `invert` argument.
  - `layer_string_lookup()` gains an `idf_weights` argument.

- Fixed issue where `input_shape` supplied to custom layers defined with `new_layer_class()`  
  would result in an error (#1338)

- New `callback_backup_and_restore()`, for resuming an interrupted `fit()` call.

- The merging family of layers (`layer_add`, `layer_concatenate`, etc.) gain the ability
  to accept layers in `...`, allowing for easier composition of residual blocks with the pipe `%>%`.
  e.g. something like this now works:
  ```r
  block_1_output <- ...
  block_2_output <- block_1_output %>% 
    layer_conv_2d(64, 3, activation = "relu", padding = "same") %>% 
    layer_add(block_1_output)
  ```
  
- `model$get_config()` method now returns an R object that can be safely serialized 
  to rds.
  
- `keras_array()` now reflects unconverted Python objects. This enables passing
  objects like `pandas.Series()` to `fit()` and `evaluate()` methods. (#1341)

# keras 2.9.0

- New functions for constructing custom keras subclasses:
  - `new_model_class()`
  - `new_layer_class()`
  - `new_callback_class()`
  - `new_metric_class()`
  - `new_loss_class()`
  - `new_learning_rate_schedule_class()`.

  Also provided is `mark_active()`, a decorator for indicating a class method
  should be an active binding (i.e., decorated with Python's `@property`).
  `mark_active()` can be used in the `new_*_class` family of class constructors
  as well as `%py_class%`.

-  `r_to_py()` method for R6 classes and `%py_class%` gain support for
  `private` fields and methods. Any R objects stored in `private` will only be
  available to methods, and will not be converted to Python.

- New family of functions for controlling optimizer learning rates during training:
  -  `learning_rate_schedule_cosine_decay()`
  -  `learning_rate_schedule_cosine_decay_restarts()`
  -  `learning_rate_schedule_exponential_decay()`
  -  `learning_rate_schedule_inverse_time_decay()`
  -  `learning_rate_schedule_piecewise_constant_decay()`
  -  `learning_rate_schedule_polynomial_decay()`

  Also, a function for constructing custom learning rate schedules:
  `new_learning_rate_schedule_class()`.

- New L2 unit normilization layer: `layer_unit_normalization()`.

- New `regularizer_orthogonal`, a regularizer that encourages
  orthogonality between the rows (or columns) or a weight matrix.

- New `zip_lists()` function for transposing lists, optionally matching by name.

- New `plot()` S3 method for models.
- `pydot` is now included in the packages installed by `install_keras()`.
- The `png` package is now listed under Suggests.

- The `%<>%` assignment pipe from magrittr is exported.

- `format()` method for keras models (and derivative methods `print()`, `summary()`,
  `str()`, and `py_str()`):
    - gain a new arg `compact`. If `TRUE` (the default) white-space only
      lines are stripped out of `model.summary()`.
    - If any layers are marked non-trainable or frozen, the model summary
      now includes a "Trainable" column, indicating if a layer is frozen.

- `freeze_weights()` and `unfreeze_weights()`:
  - gain a flexible `which` argument that can accept layer names (as character strings),
    an integer vector, a boolean vector, or a function that returns a boolean
    when called with a layer. (see updated examples in `?freeze_weights`
  - `from` and `to` arguments gain the ability to accept negative integers,
     to specify layers counting from the end of the layers list.

- `get_weights()` gains a `trainable` argument that can accept `TRUE` or `FALSE`,
  allowing for returning only the unfrozen or frozen weights, respectively.

- `timeseries_dataset_from_array()`:
    - R arrays are now cast to the floatx dtype ("float32" by default)
    - `start_index` and `end_index` now are 1-based.

- `image_dataset_from_directory()` gains a `crop_to_aspect_ratio` argument which
  can be used to prevent distorting images when resizing to a new aspect ratio.

- `Layer` is deprecated, superseded by `new_layer_class()`.

- `load_model_tf()` argument `custom_objects` gains the ability to accept an
  unnamed list (e.g, of objects returned by `new_layer_class()` or similar).
  Appropriate names for the supplied objects are automatically inferred.

- Fixed an issue where negative values less than -1 supplied to `axis`
  arguments were selecting the wrong axis.

- `get_layer()` gains the ability to accept negative values for the `index` argument.

- Fixed warning from `create_layer_wrapper()` when the custom layer didn't have
  an overridden `initialize` or `__init__` method.

- Backend functions:
  - k_clip() `min_value` and `max_value` gain default values of `NULL`,
    can be omitted. `NULL` is taken as -Inf or Inf, respectively.
  - k_squeeze(): `axis` argument can be omitted, in which case all axes of size 1 are dropped.
  - k_tile(): `n` argument can now be supplied as a tensor.
  - New function `k_unstack()`.

- KerasTensor objects (e.g, returned by `layer_input()`) now inherit S3 methods
  for `"tensorflow.tensor"`.

- `plot.keras_training_history()` no longer issues message
  ``` `geom_smooth()` using formula 'y ~ x' ``` when `method = "ggplot2"`.

- `print` and related methods for models (`format`, `summary`) now accept
   a `width` argument.

- `evaluate()`, `fit()`, and `predict()` methods for keras Models now default
  to `verbose = "auto"`, with verbosity adjusted appropriately based on calls to
  `keras$utils$disable_interactive_logging()`, and contexts like
  `ParameterServerStrategy`.

- `install_keras()` now accepts `version = "release-cpu"` as a valid specification.

# keras 2.8.0

- Breaking change: The semantics of passing a named list to `keras_model()` have changed.

  Previously, `keras_model()` would `unname()` supplied `inputs` and `outputs`.
  Then, if a named list was passed to subsequent
  `fit()`/`evaluate()`/`call()`/`predict()` invocations,
  matching of `x` and `y` was done to the model's input and outpt `tensor$name`'s.
  Now, matching is done to `names()` of `inputs` and/or `outputs` supplied to `keras_model()`.
  Call `unname()` on `inputs` and `outputs` to restore the old behavior, e.g.:
    ```
    keras_model(unname(inputs), unname(outputs))
    ```

  `keras_model()` can now accept a named list for multi-input and/or multi-output
  models. The named list is converted to a `dict` in python.
   (Requires Tensorflow >= 2.4, Python >= 3.7).

  If `inputs` is a named list:
    - `call()`, `fit()`, `evaluate()`, and `predict()` methods can also
    accept a named list for `x`, with names matching to the
    names of `inputs` when the model was constructed.
    Positional matching of `x` is still also supported (requires python 3.7+).

  If `outputs` is a named list:
    - `fit()` and `evaluate()` methods can *only*
    accept a named list for `y`, with names matching to the
    names of `outputs` when the model was constructed.

- New layer `layer_depthwise_conv_1d()`.

- Models gain `format()` and `print()` S3 methods for compatibility
  with the latest reticulate. Both are powered by `model$summary()`.

- `summary()` method for Models gains arguments `expand_nested` and `show_trainable`,
  both default to `FALSE`.

- `keras_model_custom()` is soft deprecated. Please define custom models by
  subclassing `keras$Model` directly using `%py_class%` or `R6::R6Class()`.

- Fixed warning issued by `k_random_binomial()`.

- Fixed error raised when `k_random_binomial()` was passed a non-floating dtype.

- Added `k_random_bernouli()` as an alias for `k_random_binomial()`.

- `image_load()` gains a `color_mode` argument.

- Fixed issue where `create_layer_wrapper()` would not include arguments
  with a `NULL` default value in the returned wrapper.

- Fixed issue in `r_to_py.R6ClassGenerator` (and `%py_class%`) where
  single-expression `initialize` functions defined without `{` would error.

- Deprecated functions are no longer included in the package documentation index.

# keras 2.7.0

- Default Tensorflow + Keras version is now 2.7.

- New API for constructing RNN (Recurrent Neural Network) layers. This is a
  flexible interface that complements the existing RNN layers. It is primarily
  intended for advanced / research applications, e.g, prototyping novel
  architectures. It allows you to compose a RNN with a custom "cell", a Keras layer that
  processes one step of a sequence.
  New symbols:
    - `layer_rnn()`, which can compose with builtin cells:
    - `layer_gru_cell()`
    - `layer_lstm_cell()`
    - `layer_simple_rnn_cell()`
    - `layer_stacked_rnn_cells()`
  To learn more, including how to make a custom cell layer, see the new vignette:
  "Working with RNNs".

- New dataset functions:
  - `text_dataset_from_directory()`
  - `timeseries_dataset_from_array()`

- New layers:
  - `layer_additive_attention()`
  - `layer_conv_lstm_1d()`
  - `layer_conv_lstm_3d()`

- `layer_cudnn_gru()` and `layer_cudnn_lstm()` are deprecated.
  `layer_gru()` and `layer_lstm()` will automatically use CuDNN if it is available.

- `layer_lstm()` and `layer_gru()`:
  default value for `recurrent_activation` changed
  from `"hard_sigmoid"` to `"sigmoid"`.

- `layer_gru()`: default value `reset_after` changed from `FALSE` to `TRUE`

- New vignette: "Transfer learning and fine-tuning".

- New applications:
  - MobileNet V3: `application_mobilenet_v3_large()`, `application_mobilenet_v3_small()`
  - ResNet: `application_resnet101()`, `application_resnet152()`, `resnet_preprocess_input()`
  - ResNet V2:`application_resnet50_v2()`, `application_resnet101_v2()`,
              `application_resnet152_v2()` and `resnet_v2_preprocess_input()`
  - EfficientNet: `application_efficientnet_b{0,1,2,3,4,5,6,7}()`

- Many existing `application_*()`'s gain argument `classifier_activation`,
  with default `'softmax'`.
  Affected: `application_{xception, inception_resnet_v2, inception_v3, mobilenet, vgg16, vgg19}()`

- New function `%<-active%`, a ergonomic wrapper around `makeActiveBinding()`
  for constructing Python `@property` decorated methods in `%py_class%`.

- `bidirectional()` sequence processing layer wrapper gains a `backwards_layer` arguments.

- Global pooling layers `layer_global_{max,average}_pooling_{1,2,3}d()` gain a
  `keepdims` argument with default value `FALSE`.

- Signatures for layer functions are in the process of being simplified.
  Standard layer arguments are moving to `...` where appropriate
  (and will need to be provided as named arguments).
  Standard layer arguments include:
    `input_shape`, `batch_input_shape`, `batch_size`, `dtype`,
    `name`, `trainable`, `weights`.
  Layers updated:
    `layer_global_{max,average}_pooling_{1,2,3}d()`,
    `time_distributed()`, `bidirectional()`,
    `layer_gru()`, `layer_lstm()`, `layer_simple_rnn()`

- All the backend function with a shape argument `k_*(shape =)` that now accept a
  a mix of integer tensors and R numerics in the supplied list.

- All layer functions now accept `NA` as a synonym for `NULL` in arguments
  that specify shape as a vector of dimension values,
  e.g., `input_shape`, `batch_input_shape`.

- `k_random_uniform()` now automatically casts `minval` and `maxval` to the output dtype.

- `install_keras()` gains arg with default `pip_ignore_installed = TRUE`.

# keras 2.6.1

- New family of *preprocessing* layers. These are the spiritual successor to the `tfdatasets::step_*` family of data transformers (to be deprecated in a future release).
  Added a new vignette: "Working with Preprocessing Layers".
  New functions:

  Image preprocessing:
    - `layer_resizing()`
    - `layer_rescaling()`
    - `layer_center_crop()`

  Image augmentation:
    - `layer_random_crop()`
    - `layer_random_flip()`
    - `layer_random_translation()`
    - `layer_random_rotation()`
    - `layer_random_zoom()`
    - `layer_random_contrast()`
    - `layer_random_height()`
    - `layer_random_width()`

  Categorical features preprocessing:
    - `layer_category_encoding()`
    - `layer_hashing()`
    - `layer_integer_lookup()`
    - `layer_string_lookup()`

  Numerical features preprocessing:
    - `layer_normalization()`
    - `layer_discretization()`

  These join the previous set of text preprocessing functions, each of which have some minor changes:
    - `layer_text_vectorization()` (changed arguments)
    - `get_vocabulary()`
    - `set_vocabulary()`
    - `adapt()`

- `adapt()` changes:
  - Now accepts all *features preprocessing* layers, previously
    only `layer_text_vectorization()` instances were valid.
  - `reset_state` argument is removed. It only ever accepted the default value of `TRUE`.
  - New arguments `batch_size` and `steps`.
  - Now returns the adapted layer invisibly for composability with `%>%` (previously returned `NULL`)

- `get_vocabulary()` gains a `include_special_tokens` argument.
- `set_vocabulary()`:
  - Now returns the adapted layer invisibly for composability with `%>%` (previously returned `NULL`)
  - Signature simplified. Deprecated arguments (`df_data` `oov_df_value`) are now subsumed in `...`.

- `layer_text_vectorization()`:
  - valid values for argument `output_mode` change: `"binary"` is renamed to `"multi_hot"` and
    `"tf-idf"` is renamed to `"tf_idf"` (backwards compatibility is preserved).
  - Fixed an issue where valid values of `output_mode = "int"` would incorrectly
    return a ragged tensor output shape.


- Existing layer instances gain the ability to be added to sequential models via a call. E.g.:
  ```r
  layer <- layer_dense(units = 10)
  model <- keras_model_sequential(input_shape = c(1,2,3)) %>%
    layer()
  ```

- Functions in the *merging layer* family gain the ability to return a layer instance if
  the first argument `inputs` is missing. (affected: `layer_concatenate()`, `layer_add()`,
  `layer_subtract()`, `layer_multiply()`, `layer_average()`, `layer_maximum()`,
  `layer_minimum()` ,  `layer_dot()`)

- `%py_class%` gains the ability to delay initializing the Python session until first use.
  It is now safe to implement and export `%py_class%` objects in an R package.

- Fixed an issue in `layer_input()` where passing a tensorflow `DType` objects to argument `dtype` would throw an error.

- Fixed an issue in `compile()` where passing an R function via an in-line
  call would result in an error from subsequent `fit()` calls.
  (e.g., `compile(loss = function(y_true, y_pred) my_loss(y_true, y_pred))`
  now succeeds)

- `clone_model()` gains a `clone_function` argument that allows you to customize each layer as it is cloned.

- Bumped minimum R version to 3.4. Expanded CI to test on all supported R version. Fixed regression that prevented package installation on R <= 3.4

# keras 2.6.0

Breaking changes (Tensorflow 2.6):
- Note: The following breaking changes are specific to Tensorflow version 2.6.0.
  However, the keras R package maintains compatibility with multiple versions of Tensorflow/Keras.
  You can upgrade the R package and still preserve the previous behavior by
  installing a specific version of Tensorflow: `keras::install_keras(tensorflow="2.4.0")`

- `predict_proba()` and `predict_classes()` were removed.
- `model_to_yaml()` and `model_from_yaml()` were removed.
- default changed: `layer_text_vectorization(pad_to_max_tokens=FALSE)`
- `set_vocabulary()` arguments `df_data` and `oov_df_value` are removed. They are replaced by the new argument `idf_weights`.

New Features:

- Default Tensorflow/Keras version is now 2.6

- Introduced `%py_class%`, an R-language constructor for Python classes.

- New vignettes:
  - Subclassing Python classes: How to use `%py_class%`.
  - Making new layers and models via subclassing.
  - Customizing what happens in fit (example of how to define a model, like a GAN, with a custom train step).
  - Writing your own callbacks.

- The `keras` Python module is exported

- Major changes to the underlying handling of custom R6 layer classes.
  - A new `r_to_py()` method is provided for `R6ClassGenerator` objects.
  - R6 custom layers can now inherit directly from Python layer classes
    or other R6 custom layer classes.
  - Custom R6 layers can now be instantiated directly after conversion of the class generator with `r_to_py()`, without going through `create_layer()`.
  - `KerasLayer` is deprecated (new classes should inherit directly from `keras$layers$Layer`).
  - `KerasWrapper` is deprecated (new classes should inherit directly from `keras$layers$Wrapper`).
  - `create_wrapper()` is deprecated (no longer needed, use `create_layer()` directly).
  - All layer class methods provided as R functions now have a `super` in scope that resolves to the Python super class object.
  - Methods of `super` can be accessed in the 3 common ways:
    - (Python 3 style): `super()$"__init__"()`
    - (Python 2 style): `super(ClassName, self)$"__init__"()`
    - (R6 style): `super$initialize()`
  - User defined custom classes that inherit from a Python type are responsible for calling ```super()$`__init__`(...)``` if appropriate.
  - Custom layers can now properly handle masks (#1225)
    - `supports_masking = TRUE` attribute is now supported
    - `compute_mask()` user defined method is now supported
  - `call()` methods now support a `training` argument, as well as any additional arbitrary user-defined arguments

- `Layer()` custom layer constructor is now lazy about initializing the Python session and safe to use on the top level of an R package (#1229).

- New function `create_layer_wrapper()` that can create a composing R function wrapper around a custom layer class.

- Refactored `install_keras()` (along with `tensorflow::install_tensorflow()`).
  Installation should be more reliable for more users now.
  If you encounter installation issues, please file an issue: https://github.com/rstudio/keras/issues/new
  - Potentially breaking change: numeric versions supplied without a patchlevel now automatically pull the latest patch release.
    (e.g. `install_keras(tensorflow="2.4")` will install tensorflow version "2.4.2". Previously it would install "2.4.0")

  - pandas is now a default extra packages installed by `install_keras()`
  - pyyaml is no longer a installed by default if the Tensorflow version >= 2.6.

- Loss functions:
  - All the loss functions gain the ability to return a callable
    (a `keras$losses$Loss` instance) if `y_true` and `y_pred` arguments are missing.
  - New builtin loss functions:

      -  `loss_huber()`
      -  `loss_kl_divergence()`

- Metric functions:
  - All the metric functions gain the ability to return a `keras$metrics$Metric` instance if called without `y_true` and `y_pred`
  - Each metric function is now documented separately, with a common `?Metric` topic demonstrating example usage.
  - New built-in metrics:

      -  `metric_true_negatives()`
      -  `metric_true_positives()`
      -  `metric_false_negatives()`
      -  `metric_false_positives()`
      -  `metric_specificity_at_sensitivity()`
      -  `metric_sensitivity_at_specificity()`
      -  `metric_precision()`
      -  `metric_precision_at_recall()`
      -  `metric_sum()`
      -  `metric_recall()`
      -  `metric_recall_at_precision()`
      -  `metric_root_mean_squared_error()`
      -  `metric_sparse_categorical_accuracy()`
      -  `metric_mean_tensor()`
      -  `metric_mean_wrapper()`
      -  `metric_mean_iou()`
      -  `metric_mean_relative_error()`
      -  `metric_logcosh_error()`
      -  `metric_mean()`
      -  `metric_cosine_similarity()`
      -  `metric_categorical_hinge()`
      -  `metric_accuracy()`
      -  `metric_auc()`

- `keras_model_sequential()` gains the ability to accept arguments that
  define the input layer like `input_shape` and `dtype`.
  See `?keras_model_sequential` for details and examples.

- Many layers gained new arguments, coming to parity with the interface
  available in the latest Python version:

    | layer name                   | new argument     |
    |------------------------------|------------------|
    | `layer_gru`                  | `time_major`     |
    | `layer_lstm`                 | `time_major`     |
    | `layer_max_pooling_1d`       | `data_format`    |
    | `layer_conv_lstm_2d`         | `return_state`   |
    | `layer_depthwise_conv_2d`    | `dilation_rate`  |
    | `layer_conv_3d_transpose`    | `dilation_rate`  |
    | `layer_conv_1d`              | `groups`         |
    | `layer_conv_2d`              | `groups`         |
    | `layer_conv_3d`              | `groups`         |
    | `layer_locally_connected_1d` | `implementation` |
    | `layer_locally_connected_2d` | `implementation` |
    | `layer_text_vectorization`   | `vocabulary`     |


- The `compile()` method for keras models has been updated:
  - `optimizer` is now an optional argument. It defaults to `"rmsprop"` for regular keras models.
     Custom models can specify their own default optimizer.
  - `loss` is now an optional argument.
  - New optional arguments: `run_eagerly`, `steps_per_execution`.
  - `target_tensors` and `sample_weight_mode` must now be supplied as named arguments.

- Added activation functions swish and gelu. (#1226)

- `set_vocabulary()` gains a `idf_weights` argument.

- All optimizer had argument `lr` renamed to `learning_rate`.
  (backwards compatibility is preserved, an R warning is now issued).

- The glue package was added to Imports

- Refactored automated tests to closer match the default installation procedure
  and compute environment of most user.

- Expanded CI test coverage to include R devel, oldrel and 3.6.


# keras 2.4.0

- Use compat module when using `set_session` and `get_session`. (#1046)
- Allows passing other arguments to `keras_model` eg `name`. (#1045)
- Fixed bug when serializing models with the plaidml backends.(#1084)
- Install keras no longer tries to install scipy because it's already installed by tensorflow (#1081)
- Fixed bug with `layer_text_vectorization` with TensorFlow >= 2.3 (#1131)
- Handle renamed argument `text` to `input_text` in `text_one_hot` (#1133)
- Added TensorFlow 2.3 to the CI (#1102)
- Fix C stack error when using Image Data Generators and Time Series generators with TensorFlow <= 2.0.1 (#1135)
- Fixed warning raised in the initial epoch (@gsteinbu #1130)
- Consistent result when using `text_hashing_trick` with missing values (@topepo #1048)
- Added a custom error message for `k_logsumexp` as it was removed from Keras (#1137)
- Fixed bug when printing models that are not built yet. (#1138)
- Fix drop_duplicates DeprecationWarning with tf 2.3 (@gsteinbu #1139 #1141)
- Fixed bug when plotting the model history if the model used an early stopping callback (#1140)
- `install_keras` now installs a fixed version of h5py, because newer versions are backward incompatible. (#1142)
- Simplify testing utilities by using a `helper-*` file. (#1173)
- Deprecated `hdf5_matrix` if using TF >= 2.4 (#1175)
- Fixed TensorFlow nightly installation on CI (#1176)
- Support for TensorFlow v2.4: just small fixes for custom classes. (#1177)
- Added `untar` argument to `get_file` as it seems to be slightly different from `extract` (#1179)
- Warn when not using the tensorflow implementation of Keras (#1181)
- Added `layer_layer_normalization` (#1183)
- Added `layer_multihead_attention` (#1184)
- Added `image_dataset_from_directory` (#1185)
- Fixed bug when using a custom layer with a time distributed adverb. (#1188)
- Added the `ragged` argument to `layer_input`. (#1193)
- Fixed `*_generator` deadlocks with recent versions of TensorFlow (#1197)

# Keras 2.2.3.0 (CRAN)

- Added `layer_attention` (#1000) by @atroiano.
- Fixed issue regarding the KerasMetricsCallback with TF v2.2 (#1020)

# Keras 2.2.5.0 (CRAN)

- Added `layer_dense_features`.

- Added `on_test_*`, `on_test_batch_*`, `on_predict_*` and `on_predict_*` to callback options.

- Search for the right optimizers and initializers on TensorFlow 2.0

- Fixed bug in function generators when using models with multiple inputs. (#740)

- Added `export_savedmodel` support for TensorFlow 2.0 (#773)

- Fixed bug when using `metric_` functions. (#804)

- Allow users to pass additional arguments to `install_keras` (#808)

- Enabled calling Keras models with R arrays. (#806)

- Allow passing `data.frames` as inputs to Keras models. (#822)

- Fixed bug when passing a fixed validation set to `fit_generator` (#837)

- Fixed bug when passing a TensorFlow dataset to `fit` within a `tf$distribute` scope. (#856)

- `install_keras` will now install Keras dependencies (#856). It won't re-install TensorFlow if it's already installed.

- Fixed deprecation messages showed with TensorFlow v1.14.

- Largely reduced tests verbosity.

## Keras 2.2.4.1 (CRAN)

- Use `tf.keras` as default implementation module.

- Added AppVeyor to test on Windows.

- Added `flow_images_from_dataframe` function (#658).

- Allow for unknown `input_shape` in `application_*` functions.

- Added `save_model_tf` and `load_model_tf` to save/load models in the TensorFlow's
SavedModel format.


# Keras 2.2.4 (CRAN)

- Improve handling of `timeseries_generator()` in calls to `fit_generator()`

- Add support for `input_shape` argument to `layer_dropout()`

- Improve error message for data frames passed to `fit()`, etc.

- Use 1-based axis indices for `k_gather()`

- Added `version` parameter to `install_keras()` for installing alternate/older versions

- Added `activation_exponential()` function.

- Added `threshold` parameter to `activation_relu()`

- Added `restore_best_weights` parameter to `callback_model_checkpoint()`

- Added `update_freq` parameter to `callback_tensorboard()`

- Added `negative_slope` and `threshold` parameters to `layer_activation_relu()`

- Added `output_padding` and `dilation_rate` parameters to `layer_conv_2d_transpose()`

- Added `output_padding` argument to `layer_conv_3d_transpose()`

- Added `data_format` argument to `layer_separable_conv_1d()`, `layer_average_pooling_1d()`,
  `layer_global_max_pooling_1d()`, and `layer_global_average_pooling_1d()`

- Added `interpolation` argument to `layer_upsampling_1d()` and `layer_upsampling_2d()`

- Added `dtype` argument to `to_categorical()`

- Added `layer_activation_selu()` function.

- Added `KerasWrapper` class and corresponding `create_wrapper` function.


# Keras 2.2.0

- Fix issue with serializing models that have constraint arguments

- Fix issue with `k_tile` that needs an integer vector instead of a list as the `n` argument.

- Fix issue with user-supplied `output_shape` in `layer_lambda()` not being supplied to tensorflow backends

- Filter out metrics that were created for callbacks (e.g. `lr`)

- Added `application_mobilenet_v2()` pre-trained model

- Added `sample_weight` parameter to `flow_images_from_data()`

- Use native Keras implementation (rather than SciPy) for `image_array_save()`

- Default `layer_flatten()` `data_format` argument to `NULL` (which defaults to global Keras config).

- Add `baseline` argument to `callback_early_stopping()` (stop training if a given baseline isn't reached).

- Add `data_format` argument to `layer_conv_1d()`.

- Add `layer_activation_relu()`, making the ReLU activation easier to configure
  while retaining easy serialization capabilities.

- Add `axis = -1` argument in backend crossentropy functions specifying the class prediction
  axis in the input tensor.

- Handle symbolic tensors and TF datasets in calls to `fit()`, `evaluate()`, and `predict()`

- Add `embeddings_data` argument to `callback_tensorboard()`

- Support for defining custom Keras models (i.e. custom `call()` logic for forward pass)

- Handle named list of model output names in `metrics` argument of `compile()`

- New `custom_metric()` function for defining custom metrics in R

- Provide typed wrapper for categorical custom metrics

- Provide access to Python layer within R custom layers

- Don't convert custom layer output shape to tuple when shape is a list
  or tuple of other shapes

- Re-export `shape()` function from tensorflow package

- Re-export `tuple()` function from reticulate package

- Indexes for `get_layer()` are now 1-based (for consistency w/ `freeze_weights()`)

- Accept named list for `sample_weight` argument to `fit()`


## Keras 2.1.6

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


# Keras 2.1.2

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


# keras 2.0.9

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


# keras 2.0.8

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

- `is_keras_available()` function to probe whether the Keras Python
  package is available in the current environment.

- `as.data.frame()` S3 method for Keras training history

- Remove names from `keras_model()` inputs

- Return result of `evaluate()` as named list

- Write run metrics and evaluation data to tfruns

- Provide hint to use r-tensorflow environment when importing keras


# keras 2.0.5

- Initial CRAN release
