# keras3 (development version)

- Added S3 methods for JAX array: `str`, `as.array`, `as.double`, `as.integer`, `as.numeric`.

- Added `str` S3 method for Keras Variables.

- `layer_reshape()` can now accept `-1` as a sentinel for an automatically calculated axis size.

- Updated dependencies declared by `use_backend("jax", gpu=TRUE)`
  for compatability with `keras-hub`.

- Added training loop configuration helpers:
  `config_max_epochs()`, `config_set_max_epochs()`, `config_max_steps_per_epoch()`,
  and `config_set_max_steps_per_epoch()`. The caps can also be set via the
  `KERAS_MAX_EPOCHS` and `KERAS_MAX_STEPS_PER_EPOCH` environment variables.
  Added `config_is_nnx_enabled()` to check whether JAX NNX features are enabled.

- LoRA-enabled layers (`layer_dense()`, `layer_embedding()`, `layer_einsum_dense()`)
  gain a `lora_alpha` argument to scale the adaptation delta independently of the
  chosen rank.

- `keras_variable()` now accepts a `synchronization` argument for distributed
  strategies.

- `Layer$add_weight()` gains an `overwrite_with_gradient` option and
  layers now provide a `symbolic_call()` method.

- Transposed convolution utilities now follow the latest Keras API:
  `op_conv_transpose()` defaults `strides = 1` and the `layer_conv_*_transpose()`
  layers expose `output_padding` for precise shape control.

- `layer_torch_module_wrapper()` gains an `output_shape` argument to help Keras
  infer shapes when wrapping PyTorch modules.

- `save_model_weights()` adds a `max_shard_size` argument to split large weight
  files into manageable shards.

- Added elastic deformation utilities for images: `layer_random_elastic_transform()`
  and the lower-level `op_image_elastic_transform()`.

- Added `loss_categorical_generalized_cross_entropy()` for training with noisy
  labels.

- Added the Muon optimizer via `optimizer_muon()`.

- Added complex-valued helpers: S3 `Arg()` methods for tensors, `op_angle()`,
  and conversions `op_view_as_real()` / `op_view_as_complex()`.

- Added signal window operations: `op_bartlett()`, `op_blackman()`,
  `op_hamming()`, `op_hanning()`, and `op_kaiser()`.

- Expanded numeric operations with `op_layer_normalization()`, `op_cbrt()`,
  `op_corrcoef()`, `op_deg2rad()`, `op_heaviside()`, the new `op_sparse_sigmoid()`
  plus matching `activation_sparse_sigmoid()`, and an `attn_logits_soft_cap`
  argument for `op_dot_product_attention()`.

- `layer_layer_normalization()` removes the `rms_scaling` argument.

# keras3 1.4.0

- New `op_subset()` and `x@r[...]` methods enable tensor subsetting
  using R's `[` semantics and idioms.

- New subset assignment methods implemented for tensors:
    `op_subset(x, ...) <- value` and `x@r[...] <- value`

- Breaking changes: All operations prefixed with `op_` now return 1-based
  indices by default. The following functions that return or consume indices have
  changed:
    `op_argmax()`, `op_argmin()`, `op_top_k()`, `op_argpartition()`,
    `op_searchsorted()`, `op_argsort()`, `op_digitize()`, `op_nonzero()`,
    `op_split()`, `op_trace()`, `op_swapaxes()`, `op_ctc_decode()`,
    `op_ctc_loss()`, `op_one_hot()`, `op_arange()`

- `op_arange()` now matches the semantics of `base::seq()`. By default
  it starts, includes the end value, and automatically infers step direction.

- `op_one_hot()` now infers `num_classes` if supplied a factor.

- `op_hstack()` and `op_vstack()` now accept arguments passed via `...`.

- `application_decode_predictions()` now returns a processed data frame by
  default or a decoder function if predictions are missing.

- `application_preprocess_inputs()` returns a preprocessor function if
  inputs are missing.

- Various new examples added to documentation,
  including `op_scatter()`, `op_switch()`, and `op_nonzero()`.

- New `x@py[...]` accessor introduced for Python-style 0-based indexing of tensors.

- New `Summary` group generic method for `keras_shape`, enabling usage like
  `prod(shape(3, 4))`

- `KERAS_HOME` is now set to `tools::R_user_dir("keras3", "cache")` if
 `~/.keras` does not exist and `KERAS_HOME` is unset.

- new `op_convert_to_array()` to convert a tensor to an R array.

- Added compatibility with Keras v3.9.2.
  - New operations added:

    - `op_rot90()`
    - `op_rearrange()` (Einops-style)
    - `op_signbit()`
    - `op_polar()`
    - `op_image_perspective_transform()`
    - `op_image_gaussian_blur()`

  - New layers introduced:

    - `layer_rms_normalization()`
    - `layer_aug_mix()`
    - `layer_cut_mix()`
    - `layer_random_invert()`
    - `layer_random_erasing()`
    - `layer_random_gaussian_blur()`
    - `layer_random_perspective()`

  - `layer_resizing()` gains an `antialias` argument.

  - `keras_input()`, `keras_model_sequential()`, and `op_convert_to_tensor()` gain a `ragged` argument.

  - `layer$pop_layer()` gains a `rebuild` argument and now returns the removed layer.

  - New `rematerialized_call()` method added to `Layer` objects.

  - Documentation improvements and minor fixes.

- Fixed an issue where `op_shape()` would sometimes return a TensorFlow `TensorShape`

- Fixes for `metric_iou()`, `op_top_k()`, and `op_eye()` being called with R atomic doubles


# keras3 1.3.0

- Keras now uses `reticulate::py_require()` to resolve Python dependencies.
  Calling `install_keras()` is no longer required (but is still supported).

- `use_backend()` gains a `gpu` argument, to specify if a GPU-capable set of
  dependencies should be resolved by `py_require()`.

- The progress bar in `fit()`, `evaluate()` and `predict()` now
  defaults to not presenting during testthat tests.

- `dotty::.` is now reexported.

- `%*%` now dispatches to `op_matmul()` for tensorflow tensors, which
  has relaxed shape constraints compared to `tf$matmul()`.

- Fixed an issue where calling a `Metric` and `Loss` object
  with unnamed arguments would error.

## Added compatibility with Keras v3.8.0. User-facing changes:

- New symbols:
  - `activation_sparse_plus()`
  - `activation_sparsemax()`
  - `activation_threshold()`
  - `layer_equalization()`
  - `layer_mix_up()`
  - `layer_rand_augment()`
  - `layer_random_color_degeneration()`
  - `layer_random_color_jitter()`
  - `layer_random_grayscale()`
  - `layer_random_hue()`
  - `layer_random_posterization()`
  - `layer_random_saturation()`
  - `layer_random_sharpness()`
  - `layer_random_shear()`
  - `op_diagflat()`
  - `op_sparse_plus()`
  - `op_sparsemax()`
  - `op_threshold()`
  - `op_unravel_index()`

- Add argument axis to tversky loss
- New: ONNX model export with `export_savedmodel()`
- Doc improvements and bug fixes.
- JAX specific changes: Add support for JAX named scope
- TensorFlow specific changes: Make `random_shuffle()` XLA compilable


## Added compatibility with Keras v3.7.0. User-facing changes:

### New functions

#### Activations
- `activation_celu()`
- `activation_glu()`
- `activation_hard_shrink()`
- `activation_hard_tanh()`
- `activation_log_sigmoid()`
- `activation_soft_shrink()`
- `activation_squareplus()`
- `activation_tanh_shrink()`

#### Configuration
- `config_disable_flash_attention()`
- `config_enable_flash_attention()`
- `config_is_flash_attention_enabled()`

#### Layers and Initializers
- `initializer_stft()`
- `layer_max_num_bounding_boxes()`
- `layer_stft_spectrogram()`

#### Losses and Metrics
- `loss_circle()`
- `metric_concordance_correlation()`
- `metric_pearson_correlation()`

#### Operations
- `op_celu()`
- `op_exp2()`
- `op_glu()`
- `op_hard_shrink()`
- `op_hard_tanh()`
- `op_ifft2()`
- `op_inner()`
- `op_soft_shrink()`
- `op_squareplus()`
- `op_tanh_shrink()`

#### New arguments

* `callback_backup_and_restore()`: Added `double_checkpoint` argument to save a fallback checkpoint
* `callback_tensorboard()`: Added support for `profile_batch` argument
* `layer_group_query_attention()`: Added `flash_attention` and `seed` arguments
* `layer_multi_head_attention()`: Added `flash_attention` argument
* `metric_sparse_top_k_categorical_accuracy()`: Added `from_sorted_ids` argument

### Performance improvements

* Added native Flash Attention support for GPU (via cuDNN) and TPU (via Pallas kernel) in JAX backend
* Added opt-in native Flash Attention support for GPU in PyTorch backend
* Enabled additional kernel fusion via bias_add in TensorFlow backend
* Added support for Intel XPU devices in PyTorch backend


- `install_keras()` changes: if a GPU is available, the default is now to
  install a CPU build of TensorFlow and a GPU build of JAX. To use a GPU in the
  current session, call `use_backend("jax")`.

## Added compatibility with Keras v3.6.0. User-facing changes:

#### Breaking changes:

- When using `get_file()` with `extract = TRUE` or `untar = TRUE`, the return value
  is now the path of the extracted directory, rather than the path of the archive.

#### Other changes and additions:

- Logging is now asynchronous in `fit()`, `evaluate()`, and `predict()`. This
  enables 100% compact stacking of `train_step` calls on accelerators (e.g. when
  running small models on TPU).
  - If you are using custom callbacks that rely on `on_batch_end`, this will
    disable async logging. You can re-enable it by adding
    `self$async_safe <- TRUE` to your callbacks. Note that the TensorBoard
    callback is not considered async-safe by default. Default callbacks like the
    progress bar are async-safe.

- New bitwise operations:
  - `op_bitwise_and()`
  - `op_bitwise_invert()`
  - `op_bitwise_left_shift()`
  - `op_bitwise_not()`
  - `op_bitwise_or()`
  - `op_bitwise_right_shift()`
  - `op_bitwise_xor()`

- New math operations:
  - `op_logdet()`
  - `op_trunc()`
  - `op_histogram()`

- New neural network operation: `op_dot_product_attention()`

- New image preprocessing layers:
  - `layer_auto_contrast()`
  - `layer_solarization()`

- New Model functions `get_state_tree()` and `set_state_tree()`, for retrieving
  all model variables, including trainable, non-trainable, optimizer variables,
  and metric variables.

- New `layer_pipeline()` for composing a sequence of layers. This class is useful
  for building a preprocessing pipeline. Compared to a `keras_model_sequential()`,
  `layer_pipeline()` has a few key differences:
  - It's not a Model, just a plain layer.
  - When the layers in the pipeline are compatible with `tf.data`, the pipeline
    will also remain `tf.data` compatible, regardless of the backend you use.

- New argument: `export_savedmodel(verbose = )`
- New argument: `op_normalize(epsilon = )`

- Various documentation improvements and bug fixes.


# keras3 1.2.0

- Added compatibility with Keras v3.5.0. User facing changes:

  - New functions:
    - `op_associative_scan()`
    - `op_searchsorted()`
    - `optimizer_lamb()`
  - `keras$DTypePolicy` instances can now be supplied to `dtype` argument for
    losses, metrics, and layers.
  - Add integration with the Hugging Face Hub. You can now save models to
    Hugging Face Hub directly `save_model()` and load .keras models directly
    from Hugging Face Hub with `load_model()`.
  - Added compatibility with NumPy 2.0.
  - Improved `keras$distribution` API support for very large models.
  - Bug fixes and performance improvements.
  - Add `data_format` argument to `layer_zero_padding_1d()` layer.
  - Miscellaneous documentation improvements.
  - Bug fixes and performance improvements.


# keras3 1.1.0

- Fixed issue where GPUs would not be found when running on Windows under WSL Linux.
  (reported in #1456, fixed in #1459)

- `keras_shape` objects (as returned by `keras3::shape()`) gain `==` and `!=` methods.

- Fixed warning from `tfruns::training_run()` being unable to log optimizer learning rate.

- Added compatibility with Keras v3.4.1 (no R user facing changes).

- Added compatibility with Keras v3.4.0. User facing changes:

  - New functions:
    - `op_argpartition()`
    - `op_map()`
    - `op_scan()`
    - `op_switch()`
    - `op_dtype()`
    - `op_lstsq()`
    - `op_image_hsv_to_rgb()`
    - `op_image_rgb_to_hsv()`

  - Changes:
    - Added support for arbitrary, deeply nested input/output structures in
      Functional models  (e.g. lists of lists of lists of inputs or outputs...)
    - Add support for `optional` Functional inputs.
      - `keras_input()` gains an `optional` argument.
      - `keras_model_sequential()` gains a `input_optional` argument.
    - Add support for `float8` inference for `Dense` and `EinsumDense` layers.
    - Enable `layer_feature_space()` to be used in a `{tfdatasets}` pipeline even
      when the backend isn't TensorFlow.
    - `layer_string_lookup()` can now take `tf$SparseTensor()` as input.
    - `layer_string_lookup()` returns `"int64"` dtype by default in more modes now.
    - `Layer()` instances gain attributes `path` and `quantization_mode`.
    - `Metric()$variables` is now recursive.
    - Add `training` argument to `Model$compute_loss()`.
    - `split_dataset()` now supports nested structures in dataset.
    - All applications gain a `name` argument, accept a custom name.
    - `layer_multi_head_attention()` gains a `seed` argument.
    - All losses gain a `dtype` argument.
    - `loss_dice()` gains an `axis` argument.
    - `op_ctc_decode()`, new default for `mask_index = 0`
    - All `op_image_*` functions now use default `data_format` value
      to `config_image_data_format()`
    - `op_isclose()` gains arguments `rtol`, `atol`, `equal_nan`.
    - `save_model()` gains argument `zipped`.
    - Bugs fixes and performance improvements.

# keras3 1.0.0

- Chains of `layer_*` calls with `|>` now instantiate layers in the
  same order as `%>%` pipe chains: left-hand-side first (#1440).

- `iterate()`, `iter_next()` and `as_iterator()` are now reexported from reticulate.


User facing changes with upstream Keras v3.3.3:

- new functions: `op_slogdet()`, `op_psnr()`

- `clone_model()` gains new args: `call_function`, `recursive`
  Updated example usage.

- `op_ctc_decode()` strategy argument has new default: `"greedy"`.
  Updated docs.

- `loss_ctc()` default name fixed, changed to `"ctc"`

User facing changes with upstream Keras v3.3.2:

- new function: `op_ctc_decode()`
- new function: `op_eigh()`
- new function: `op_select()`
- new function: `op_vectorize()`
- new function: `op_image_rgb_to_grayscale()`
- new function: `loss_tversky()`

- new args: `layer_resizing(pad_to_aspect_ratio, fill_mode, fill_value)`
- new arg: `layer_embedding(weights)` for providing an initial weights matrix

- new args: `op_nan_to_num(nan, posinf, neginf)`
- new args: `op_image_resize(crop_to_aspect_ratio, pad_to_aspect_ratio, fill_mode, fill_value)`
- new args: `op_argmax(keepdims)` and `op_argmin(keepdims)`

- new arg: `clear_session(free_memory)` for clearing without invoking the garbage collector.

- `metric_kl_divergence()` and `loss_kl_divergence()` clip inputs
  (`y_true` and `y_pred`) to the `[0, 1]` range.

- new `Layer()` attributes: `metrics`, `dtype_policy`

- Added initial support for float8 training

- `layer_conv_*d()` layers now support LoRa

- `op_digitize()` now supports sparse tensors.

- Models and layers now return owned metrics recursively.

- Add pickling support for Keras models. (e.g., via `reticulate::py_save_object()`)
  Note that pickling is not recommended, prefer using Keras saving APIs.


# keras3 0.2.0

New functions:

  - `quantize_weights()`: quantize model or layer weights in-place. Currently,
    only `Dense`, `EinsumDense`, and `Embedding` layers are supported (which is enough to
    cover the majority of transformers today)
  - `layer_mel_spectrogram()`
  - `layer_flax_module_wrapper()`
  - `layer_jax_model_wrapper()`

  - `loss_dice()`

  - `random_beta()`
  - `random_binomial()`

  - `config_set_backend()`: change the backend after Keras has initialized.
  - `config_dtype_policy()`
  - `config_set_dtype_policy()`

  - New Ops
    - `op_custom_gradient()`
    - `op_batch_normalization()`
    - `op_image_crop()`
    - `op_divide_no_nan()`
    - `op_normalize()`
    - `op_correlate()`
    - `
  - New family of linear algebra ops
    - `op_cholesky()`
    - `op_det()`
    - `op_eig()`
    - `op_inv()`
    - `op_lu_factor()`
    - `op_norm()`
    - `op_erfinv()`
    - `op_solve_triangular()`
    - `op_svd()`

- `audio_dataset_from_directory()`, `image_dataset_from_directory()` and `text_dataset_from_directory()` gain a `verbose` argument (default `TRUE`)

- `image_dataset_from_directory()` gains `pad_to_aspect_ratio` argument (default `FALSE`)

- `to_categorical()`, `op_one_hot()`, and `fit()` can now accept R factors,
  offset them to be 0-based (reported in `#1055`).

- `op_convert_to_numpy()` now returns unconverted NumPy arrays.

- `op_array()` and `op_convert_to_tensor()` no longer error when casting R
   doubles to integer types.

- `export_savedmodel()` now works with a Jax backend.

- `Metric()$add_variable()` method gains arg: `aggregration`.
- `Layer()$add_weight()` method gains args: `autocast`, `regularizer`, `aggregation`.

- `op_bincount()`, `op_multi_hot()`, `op_one_hot()`, and `layer_category_encoding()` now support sparse tensors.

- `op_custom_gradient()` now supports the PyTorch backend

- `layer_lstm()` and `layer_gru()` gain arg `use_cudnn`, default `'auto'`.

- Fixed an issue where `application_preprocess_inputs()` would error if supplied
  an R array as input.

- Doc improvements.

# keras3 0.1.0

- The package has been rebuilt for Keras 3.0. Refer to https://blogs.rstudio.com/ai/posts/2024-05-21-keras3/ for an overview
  and https://keras3.posit.co for the current up-to-date documentation.

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
    - `rnn_cell_gru()`
    - `rnn_cell_lstm()`
    - `rnn_cell_simple()`
    - `rnn_cells_stack()`
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
  installing a specific version of Tensorflow: `keras3::install_keras(tensorflow="2.4.0")`

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
