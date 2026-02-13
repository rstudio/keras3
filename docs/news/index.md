# Changelog

## keras3 1.5.1

- `use_backend("jax")` on macOS now defaults to `gpu = FALSE`, so
  `jax-metal` is no longer selected automatically.

- JAX now registers backend tensor S3 methods for `jax.core.Tracer`
  objects, improving compatibility for traced computations.

- On Linux, `tensorflow-cpu` is no longer pinned to `2.18.*` when
  resolving Python dependencies.

## keras3 1.5.0

CRAN release: 2025-12-22

- [`register_keras_serializable()`](https://keras3.posit.co/reference/register_keras_serializable.md)
  now updates R layer wrappers to use the registered class when called.

- Numeric ops now include
  [`op_layer_normalization()`](https://keras3.posit.co/reference/op_layer_normalization.md),
  [`op_cbrt()`](https://keras3.posit.co/reference/op_cbrt.md),
  [`op_corrcoef()`](https://keras3.posit.co/reference/op_corrcoef.md),
  [`op_deg2rad()`](https://keras3.posit.co/reference/op_deg2rad.md),
  [`op_heaviside()`](https://keras3.posit.co/reference/op_heaviside.md),
  [`op_sparse_sigmoid()`](https://keras3.posit.co/reference/op_sparse_sigmoid.md),
  and
  [`activation_sparse_sigmoid()`](https://keras3.posit.co/reference/activation_sparse_sigmoid.md).
  [`op_dot_product_attention()`](https://keras3.posit.co/reference/op_dot_product_attention.md)
  gains `attn_logits_soft_cap`.

- Added signal window operations:
  [`op_bartlett()`](https://keras3.posit.co/reference/op_bartlett.md),
  [`op_blackman()`](https://keras3.posit.co/reference/op_blackman.md),
  [`op_hamming()`](https://keras3.posit.co/reference/op_hamming.md),
  [`op_hanning()`](https://keras3.posit.co/reference/op_hanning.md), and
  [`op_kaiser()`](https://keras3.posit.co/reference/op_kaiser.md).

- Added
  [`loss_categorical_generalized_cross_entropy()`](https://keras3.posit.co/reference/loss_categorical_generalized_cross_entropy.md)
  for training with noisy labels.

- LoRA-enabled layers
  ([`layer_dense()`](https://keras3.posit.co/reference/layer_dense.md),
  [`layer_embedding()`](https://keras3.posit.co/reference/layer_embedding.md),
  [`layer_einsum_dense()`](https://keras3.posit.co/reference/layer_einsum_dense.md))
  gain a `lora_alpha` argument to scale the adaptation delta
  independently of the chosen rank.

- Added complex-valued helpers: S3
  [`Arg()`](https://rdrr.io/r/base/complex.html) methods for tensors,
  [`op_angle()`](https://keras3.posit.co/reference/op_angle.md), and
  conversions
  [`op_view_as_real()`](https://keras3.posit.co/reference/op_view_as_real.md)
  /
  [`op_view_as_complex()`](https://keras3.posit.co/reference/op_view_as_complex.md).

- Added the Muon optimizer via
  [`optimizer_muon()`](https://keras3.posit.co/reference/optimizer_muon.md).

- Added elastic deformation utilities for images:
  [`layer_random_elastic_transform()`](https://keras3.posit.co/reference/layer_random_elastic_transform.md)
  and the lower-level
  [`op_image_elastic_transform()`](https://keras3.posit.co/reference/op_image_elastic_transform.md).

- Added [`as.array()`](https://rdrr.io/r/base/array.html) support for
  `PIL.Image.Image` objects.

- Transposed convolution utilities now follow the latest Keras API:
  [`op_conv_transpose()`](https://keras3.posit.co/reference/op_conv_transpose.md)
  defaults to `strides = 1`, and `layer_conv_*_transpose()` layers
  expose `output_padding` for precise shape control.

- [`register_keras_serializable()`](https://keras3.posit.co/reference/register_keras_serializable.md)
  now returns a registered Python callable, making it easier to use with
  bare R functions.

- [`save_model_weights()`](https://keras3.posit.co/reference/save_model_weights.md)
  adds a `max_shard_size` argument to split large weight files into
  manageable shards.

- [`keras_variable()`](https://keras3.posit.co/reference/keras_variable.md)
  now accepts a `synchronization` argument for distributed strategies.

- [`layer_layer_normalization()`](https://keras3.posit.co/reference/layer_layer_normalization.md)
  now omits the `rms_scaling` argument.

- Merging layers now capture `...` with tidy dots
  ([\#1525](https://github.com/rstudio/keras3/issues/1525)).

- JAX `_DimExpr` shapes now remain symbolic when combined with R double
  scalars.

- [`layer_reshape()`](https://keras3.posit.co/reference/layer_reshape.md)
  now accepts `-1` as a sentinel for an automatically calculated axis
  size.

- [`layer_torch_module_wrapper()`](https://keras3.posit.co/reference/layer_torch_module_wrapper.md)
  gains an `output_shape` argument to help Keras infer shapes when
  wrapping PyTorch modules.

- `Layer$add_weight()` gains an `overwrite_with_gradient` option, and
  layers now provide a `symbolic_call()` method.

- Added [`str()`](https://rdrr.io/r/utils/str.html) S3 method for Keras
  `Variable`s.

- JAX arrays now have S3 methods for
  [`str()`](https://rdrr.io/r/utils/str.html),
  [`as.array()`](https://rdrr.io/r/base/array.html),
  [`as.double()`](https://rdrr.io/r/base/double.html),
  [`as.integer()`](https://rdrr.io/r/base/integer.html), and
  [`as.numeric()`](https://rdrr.io/r/base/numeric.html).

- Backend tensors now support base array helpers:
  [`t()`](https://rdrr.io/r/base/t.html),
  [`aperm()`](https://rdrr.io/r/base/aperm.html), and
  [`all.equal()`](https://rdrr.io/r/base/all.equal.html).

- Added
  [`pillar::type_sum()`](https://pillar.r-lib.org/reference/type_sum.html)
  for JAX variables and `JaxVariable`;
  [`str()`](https://rdrr.io/r/utils/str.html) now covers the new JAX
  variable class.

- Added training caps via
  [`config_max_epochs()`](https://keras3.posit.co/reference/config_max_epochs.md),
  [`config_set_max_epochs()`](https://keras3.posit.co/reference/config_max_epochs.md),
  [`config_max_steps_per_epoch()`](https://keras3.posit.co/reference/config_max_epochs.md),
  and
  [`config_set_max_steps_per_epoch()`](https://keras3.posit.co/reference/config_max_epochs.md).
  The caps can also be set via the `KERAS_MAX_EPOCHS` and
  `KERAS_MAX_STEPS_PER_EPOCH` environment variables. Added
  [`config_is_nnx_enabled()`](https://keras3.posit.co/reference/config_is_nnx_enabled.md)
  to check whether JAX NNX features are enabled.

- Built-in dataset loaders now accept `convert = FALSE` to return NumPy
  arrays instead of R arrays.

- `plot(history, theme_bw = TRUE)` is now compatible with `ggplot2`
  3.4.0.

- `plot(model)` now reads DPI from `options(keras.plot.model.dpi = 200)`
  (default is 200).

- Reexported reticulate functions:
  [`py_help()`](https://rstudio.github.io/reticulate/reference/py_help.html),
  [`py_to_r()`](https://rstudio.github.io/reticulate/reference/r-py-conversion.html),
  [`r_to_py()`](https://rstudio.github.io/reticulate/reference/r-py-conversion.html),
  [`py_require()`](https://rstudio.github.io/reticulate/reference/py_require.html),
  and
  [`import()`](https://rstudio.github.io/reticulate/reference/import.html).

- `super()$initialize()` now works in subclassed Keras classes, and
  `super()` behavior is improved in subclasses.

- `use_backend("jax", gpu = TRUE)` now declares dependencies compatible
  with `keras-hub`.

- Exported
  [`named_list()`](https://keras3.posit.co/reference/named_list.md).

- Switching backends twice in a row now works reliably.

- [`layer_dropout()`](https://keras3.posit.co/reference/layer_dropout.md)
  now preserves `noise_shape` as an integer array so length-one shapes
  are passed to Keras as iterables.

## keras3 1.4.0

CRAN release: 2025-05-04

- New [`op_subset()`](https://keras3.posit.co/reference/op_subset.md)
  and `x@r[...]` methods enable tensor subsetting using R’s `[`
  semantics and idioms.

- New subset assignment methods implemented for tensors:
  `op_subset(x, ...) <- value` and `x@r[...] <- value`

- Breaking changes: All operations prefixed with `op_` now return
  1-based indices by default. The following functions that return or
  consume indices have changed:
  [`op_argmax()`](https://keras3.posit.co/reference/op_argmax.md),
  [`op_argmin()`](https://keras3.posit.co/reference/op_argmin.md),
  [`op_top_k()`](https://keras3.posit.co/reference/op_top_k.md),
  [`op_argpartition()`](https://keras3.posit.co/reference/op_argpartition.md),
  [`op_searchsorted()`](https://keras3.posit.co/reference/op_searchsorted.md),
  [`op_argsort()`](https://keras3.posit.co/reference/op_argsort.md),
  [`op_digitize()`](https://keras3.posit.co/reference/op_digitize.md),
  [`op_nonzero()`](https://keras3.posit.co/reference/op_nonzero.md),
  [`op_split()`](https://keras3.posit.co/reference/op_split.md),
  [`op_trace()`](https://keras3.posit.co/reference/op_trace.md),
  [`op_swapaxes()`](https://keras3.posit.co/reference/op_swapaxes.md),
  [`op_ctc_decode()`](https://keras3.posit.co/reference/op_ctc_decode.md),
  [`op_ctc_loss()`](https://keras3.posit.co/reference/op_ctc_loss.md),
  [`op_one_hot()`](https://keras3.posit.co/reference/op_one_hot.md),
  [`op_arange()`](https://keras3.posit.co/reference/op_arange.md)

- [`op_arange()`](https://keras3.posit.co/reference/op_arange.md) now
  matches the semantics of
  [`base::seq()`](https://rdrr.io/r/base/seq.html). By default it
  starts, includes the end value, and automatically infers step
  direction.

- [`op_one_hot()`](https://keras3.posit.co/reference/op_one_hot.md) now
  infers `num_classes` if supplied a factor.

- [`op_hstack()`](https://keras3.posit.co/reference/op_hstack.md) and
  [`op_vstack()`](https://keras3.posit.co/reference/op_vstack.md) now
  accept arguments passed via `...`.

- [`application_decode_predictions()`](https://keras3.posit.co/reference/process_utils.md)
  now returns a processed data frame by default or a decoder function if
  predictions are missing.

- [`application_preprocess_inputs()`](https://keras3.posit.co/reference/process_utils.md)
  returns a preprocessor function if inputs are missing.

- Various new examples added to documentation, including
  [`op_scatter()`](https://keras3.posit.co/reference/op_scatter.md),
  [`op_switch()`](https://keras3.posit.co/reference/op_switch.md), and
  [`op_nonzero()`](https://keras3.posit.co/reference/op_nonzero.md).

- New `x@py[...]` accessor introduced for Python-style 0-based indexing
  of tensors.

- New `Summary` group generic method for `keras_shape`, enabling usage
  like `prod(shape(3, 4))`

- `KERAS_HOME` is now set to `tools::R_user_dir("keras3", "cache")` if
  `~/.keras` does not exist and `KERAS_HOME` is unset.

- new
  [`op_convert_to_array()`](https://keras3.posit.co/reference/op_convert_to_numpy.md)
  to convert a tensor to an R array.

- Added compatibility with Keras v3.9.2.

  - New operations added:

    - [`op_rot90()`](https://keras3.posit.co/reference/op_rot90.md)
    - [`op_rearrange()`](https://keras3.posit.co/reference/op_rearrange.md)
      (Einops-style)
    - [`op_signbit()`](https://keras3.posit.co/reference/op_signbit.md)
    - [`op_polar()`](https://keras3.posit.co/reference/op_polar.md)
    - [`op_image_perspective_transform()`](https://keras3.posit.co/reference/op_image_perspective_transform.md)
    - [`op_image_gaussian_blur()`](https://keras3.posit.co/reference/op_image_gaussian_blur.md)

  - New layers introduced:

    - [`layer_rms_normalization()`](https://keras3.posit.co/reference/layer_rms_normalization.md)
    - [`layer_aug_mix()`](https://keras3.posit.co/reference/layer_aug_mix.md)
    - [`layer_cut_mix()`](https://keras3.posit.co/reference/layer_cut_mix.md)
    - [`layer_random_invert()`](https://keras3.posit.co/reference/layer_random_invert.md)
    - [`layer_random_erasing()`](https://keras3.posit.co/reference/layer_random_erasing.md)
    - [`layer_random_gaussian_blur()`](https://keras3.posit.co/reference/layer_random_gaussian_blur.md)
    - [`layer_random_perspective()`](https://keras3.posit.co/reference/layer_random_perspective.md)

  - [`layer_resizing()`](https://keras3.posit.co/reference/layer_resizing.md)
    gains an `antialias` argument.

  - [`keras_input()`](https://keras3.posit.co/reference/keras_input.md),
    [`keras_model_sequential()`](https://keras3.posit.co/reference/keras_model_sequential.md),
    and
    [`op_convert_to_tensor()`](https://keras3.posit.co/reference/op_convert_to_tensor.md)
    gain a `ragged` argument.

  - `layer$pop_layer()` gains a `rebuild` argument and now returns the
    removed layer.

  - New `rematerialized_call()` method added to `Layer` objects.

  - Documentation improvements and minor fixes.

- Fixed an issue where
  [`op_shape()`](https://keras3.posit.co/reference/op_shape.md) would
  sometimes return a TensorFlow `TensorShape`

- Fixes for
  [`metric_iou()`](https://keras3.posit.co/reference/metric_iou.md),
  [`op_top_k()`](https://keras3.posit.co/reference/op_top_k.md), and
  [`op_eye()`](https://keras3.posit.co/reference/op_eye.md) being called
  with R atomic doubles

## keras3 1.3.0

CRAN release: 2025-03-03

- Keras now uses
  [`reticulate::py_require()`](https://rstudio.github.io/reticulate/reference/py_require.html)
  to resolve Python dependencies. Calling
  [`install_keras()`](https://keras3.posit.co/reference/install_keras.md)
  is no longer required (but is still supported).

- [`use_backend()`](https://keras3.posit.co/reference/use_backend.md)
  gains a `gpu` argument, to specify if a GPU-capable set of
  dependencies should be resolved by
  [`py_require()`](https://rstudio.github.io/reticulate/reference/py_require.html).

- The progress bar in
  [`fit()`](https://generics.r-lib.org/reference/fit.html),
  [`evaluate()`](https://rdrr.io/pkg/tensorflow/man/evaluate.html) and
  [`predict()`](https://rdrr.io/r/stats/predict.html) now defaults to
  not presenting during testthat tests.

- [`dotty::.`](https://kevinushey.github.io/dotty/reference/dotty.html)
  is now reexported.

- `%*%` now dispatches to
  [`op_matmul()`](https://keras3.posit.co/reference/op_matmul.md) for
  tensorflow tensors, which has relaxed shape constraints compared to
  `tf$matmul()`.

- Fixed an issue where calling a `Metric` and `Loss` object with unnamed
  arguments would error.

### Added compatibility with Keras v3.8.0. User-facing changes:

- New symbols:
  - [`activation_sparse_plus()`](https://keras3.posit.co/reference/activation_sparse_plus.md)
  - [`activation_sparsemax()`](https://keras3.posit.co/reference/activation_sparsemax.md)
  - [`activation_threshold()`](https://keras3.posit.co/reference/activation_threshold.md)
  - [`layer_equalization()`](https://keras3.posit.co/reference/layer_equalization.md)
  - [`layer_mix_up()`](https://keras3.posit.co/reference/layer_mix_up.md)
  - [`layer_rand_augment()`](https://keras3.posit.co/reference/layer_rand_augment.md)
  - [`layer_random_color_degeneration()`](https://keras3.posit.co/reference/layer_random_color_degeneration.md)
  - [`layer_random_color_jitter()`](https://keras3.posit.co/reference/layer_random_color_jitter.md)
  - [`layer_random_grayscale()`](https://keras3.posit.co/reference/layer_random_grayscale.md)
  - [`layer_random_hue()`](https://keras3.posit.co/reference/layer_random_hue.md)
  - [`layer_random_posterization()`](https://keras3.posit.co/reference/layer_random_posterization.md)
  - [`layer_random_saturation()`](https://keras3.posit.co/reference/layer_random_saturation.md)
  - [`layer_random_sharpness()`](https://keras3.posit.co/reference/layer_random_sharpness.md)
  - [`layer_random_shear()`](https://keras3.posit.co/reference/layer_random_shear.md)
  - [`op_diagflat()`](https://keras3.posit.co/reference/op_diagflat.md)
  - [`op_sparse_plus()`](https://keras3.posit.co/reference/op_sparse_plus.md)
  - [`op_sparsemax()`](https://keras3.posit.co/reference/op_sparsemax.md)
  - [`op_threshold()`](https://keras3.posit.co/reference/op_threshold.md)
  - [`op_unravel_index()`](https://keras3.posit.co/reference/op_unravel_index.md)
- Add argument axis to tversky loss
- New: ONNX model export with
  [`export_savedmodel()`](https://rdrr.io/pkg/tensorflow/man/export_savedmodel.html)
- Doc improvements and bug fixes.
- JAX specific changes: Add support for JAX named scope
- TensorFlow specific changes: Make
  [`random_shuffle()`](https://keras3.posit.co/reference/random_shuffle.md)
  XLA compilable

### Added compatibility with Keras v3.7.0. User-facing changes:

#### New functions

##### Activations

- [`activation_celu()`](https://keras3.posit.co/reference/activation_celu.md)
- [`activation_glu()`](https://keras3.posit.co/reference/activation_glu.md)
- [`activation_hard_shrink()`](https://keras3.posit.co/reference/activation_hard_shrink.md)
- [`activation_hard_tanh()`](https://keras3.posit.co/reference/activation_hard_tanh.md)
- [`activation_log_sigmoid()`](https://keras3.posit.co/reference/activation_log_sigmoid.md)
- [`activation_soft_shrink()`](https://keras3.posit.co/reference/activation_soft_shrink.md)
- [`activation_squareplus()`](https://keras3.posit.co/reference/activation_squareplus.md)
- [`activation_tanh_shrink()`](https://keras3.posit.co/reference/activation_tanh_shrink.md)

##### Configuration

- [`config_disable_flash_attention()`](https://keras3.posit.co/reference/config_disable_flash_attention.md)
- [`config_enable_flash_attention()`](https://keras3.posit.co/reference/config_enable_flash_attention.md)
- [`config_is_flash_attention_enabled()`](https://keras3.posit.co/reference/config_is_flash_attention_enabled.md)

##### Layers and Initializers

- [`initializer_stft()`](https://keras3.posit.co/reference/initializer_stft.md)
- [`layer_max_num_bounding_boxes()`](https://keras3.posit.co/reference/layer_max_num_bounding_boxes.md)
- [`layer_stft_spectrogram()`](https://keras3.posit.co/reference/layer_stft_spectrogram.md)

##### Losses and Metrics

- [`loss_circle()`](https://keras3.posit.co/reference/loss_circle.md)
- [`metric_concordance_correlation()`](https://keras3.posit.co/reference/metric_concordance_correlation.md)
- [`metric_pearson_correlation()`](https://keras3.posit.co/reference/metric_pearson_correlation.md)

##### Operations

- [`op_celu()`](https://keras3.posit.co/reference/op_celu.md)
- [`op_exp2()`](https://keras3.posit.co/reference/op_exp2.md)
- [`op_glu()`](https://keras3.posit.co/reference/op_glu.md)
- [`op_hard_shrink()`](https://keras3.posit.co/reference/op_hard_shrink.md)
- [`op_hard_tanh()`](https://keras3.posit.co/reference/op_hard_tanh.md)
- [`op_ifft2()`](https://keras3.posit.co/reference/op_ifft2.md)
- [`op_inner()`](https://keras3.posit.co/reference/op_inner.md)
- [`op_soft_shrink()`](https://keras3.posit.co/reference/op_soft_shrink.md)
- [`op_squareplus()`](https://keras3.posit.co/reference/op_squareplus.md)
- [`op_tanh_shrink()`](https://keras3.posit.co/reference/op_tanh_shrink.md)

##### New arguments

- [`callback_backup_and_restore()`](https://keras3.posit.co/reference/callback_backup_and_restore.md):
  Added `double_checkpoint` argument to save a fallback checkpoint
- [`callback_tensorboard()`](https://keras3.posit.co/reference/callback_tensorboard.md):
  Added support for `profile_batch` argument
- [`layer_group_query_attention()`](https://keras3.posit.co/reference/layer_group_query_attention.md):
  Added `flash_attention` and `seed` arguments
- [`layer_multi_head_attention()`](https://keras3.posit.co/reference/layer_multi_head_attention.md):
  Added `flash_attention` argument
- [`metric_sparse_top_k_categorical_accuracy()`](https://keras3.posit.co/reference/metric_sparse_top_k_categorical_accuracy.md):
  Added `from_sorted_ids` argument

#### Performance improvements

- Added native Flash Attention support for GPU (via cuDNN) and TPU (via
  Pallas kernel) in JAX backend

- Added opt-in native Flash Attention support for GPU in PyTorch backend

- Enabled additional kernel fusion via bias_add in TensorFlow backend

- Added support for Intel XPU devices in PyTorch backend

- [`install_keras()`](https://keras3.posit.co/reference/install_keras.md)
  changes: if a GPU is available, the default is now to install a CPU
  build of TensorFlow and a GPU build of JAX. To use a GPU in the
  current session, call `use_backend("jax")`.

### Added compatibility with Keras v3.6.0. User-facing changes:

##### Breaking changes:

- When using
  [`get_file()`](https://keras3.posit.co/reference/get_file.md) with
  `extract = TRUE` or `untar = TRUE`, the return value is now the path
  of the extracted directory, rather than the path of the archive.

##### Other changes and additions:

- Logging is now asynchronous in
  [`fit()`](https://generics.r-lib.org/reference/fit.html),
  [`evaluate()`](https://rdrr.io/pkg/tensorflow/man/evaluate.html), and
  [`predict()`](https://rdrr.io/r/stats/predict.html). This enables 100%
  compact stacking of `train_step` calls on accelerators (e.g. when
  running small models on TPU).

  - If you are using custom callbacks that rely on `on_batch_end`, this
    will disable async logging. You can re-enable it by adding
    `self$async_safe <- TRUE` to your callbacks. Note that the
    TensorBoard callback is not considered async-safe by default.
    Default callbacks like the progress bar are async-safe.

- New bitwise operations:

  - [`op_bitwise_and()`](https://keras3.posit.co/reference/op_bitwise_and.md)
  - [`op_bitwise_invert()`](https://keras3.posit.co/reference/op_bitwise_invert.md)
  - [`op_bitwise_left_shift()`](https://keras3.posit.co/reference/op_bitwise_left_shift.md)
  - [`op_bitwise_not()`](https://keras3.posit.co/reference/op_bitwise_not.md)
  - [`op_bitwise_or()`](https://keras3.posit.co/reference/op_bitwise_or.md)
  - [`op_bitwise_right_shift()`](https://keras3.posit.co/reference/op_bitwise_right_shift.md)
  - [`op_bitwise_xor()`](https://keras3.posit.co/reference/op_bitwise_xor.md)

- New math operations:

  - [`op_logdet()`](https://keras3.posit.co/reference/op_logdet.md)
  - [`op_trunc()`](https://keras3.posit.co/reference/op_trunc.md)
  - [`op_histogram()`](https://keras3.posit.co/reference/op_histogram.md)

- New neural network operation:
  [`op_dot_product_attention()`](https://keras3.posit.co/reference/op_dot_product_attention.md)

- New image preprocessing layers:

  - [`layer_auto_contrast()`](https://keras3.posit.co/reference/layer_auto_contrast.md)
  - [`layer_solarization()`](https://keras3.posit.co/reference/layer_solarization.md)

- New Model functions
  [`get_state_tree()`](https://keras3.posit.co/reference/get_state_tree.md)
  and
  [`set_state_tree()`](https://keras3.posit.co/reference/set_state_tree.md),
  for retrieving all model variables, including trainable,
  non-trainable, optimizer variables, and metric variables.

- New
  [`layer_pipeline()`](https://keras3.posit.co/reference/layer_pipeline.md)
  for composing a sequence of layers. This class is useful for building
  a preprocessing pipeline. Compared to a
  [`keras_model_sequential()`](https://keras3.posit.co/reference/keras_model_sequential.md),
  [`layer_pipeline()`](https://keras3.posit.co/reference/layer_pipeline.md)
  has a few key differences:

  - It’s not a Model, just a plain layer.
  - When the layers in the pipeline are compatible with `tf.data`, the
    pipeline will also remain `tf.data` compatible, regardless of the
    backend you use.

- New argument: `export_savedmodel(verbose = )`

- New argument: `op_normalize(epsilon = )`

- Various documentation improvements and bug fixes.

## keras3 1.2.0

CRAN release: 2024-09-05

- Added compatibility with Keras v3.5.0. User facing changes:

  - New functions:
    - [`op_associative_scan()`](https://keras3.posit.co/reference/op_associative_scan.md)
    - [`op_searchsorted()`](https://keras3.posit.co/reference/op_searchsorted.md)
    - [`optimizer_lamb()`](https://keras3.posit.co/reference/optimizer_lamb.md)
  - `keras$DTypePolicy` instances can now be supplied to `dtype`
    argument for losses, metrics, and layers.
  - Add integration with the Hugging Face Hub. You can now save models
    to Hugging Face Hub directly
    [`save_model()`](https://keras3.posit.co/reference/save_model.md)
    and load .keras models directly from Hugging Face Hub with
    [`load_model()`](https://keras3.posit.co/reference/load_model.md).
  - Added compatibility with NumPy 2.0.
  - Improved `keras$distribution` API support for very large models.
  - Bug fixes and performance improvements.
  - Add `data_format` argument to
    [`layer_zero_padding_1d()`](https://keras3.posit.co/reference/layer_zero_padding_1d.md)
    layer.
  - Miscellaneous documentation improvements.
  - Bug fixes and performance improvements.

## keras3 1.1.0

CRAN release: 2024-07-17

- Fixed issue where GPUs would not be found when running on Windows
  under WSL Linux. (reported in
  [\#1456](https://github.com/rstudio/keras3/issues/1456), fixed in
  [\#1459](https://github.com/rstudio/keras3/issues/1459))

- `keras_shape` objects (as returned by
  [`keras3::shape()`](https://keras3.posit.co/reference/shape.md)) gain
  `==` and `!=` methods.

- Fixed warning from
  [`tfruns::training_run()`](https://rdrr.io/pkg/tfruns/man/training_run.html)
  being unable to log optimizer learning rate.

- Added compatibility with Keras v3.4.1 (no R user facing changes).

- Added compatibility with Keras v3.4.0. User facing changes:

  - New functions:
    - [`op_argpartition()`](https://keras3.posit.co/reference/op_argpartition.md)
    - [`op_map()`](https://keras3.posit.co/reference/op_map.md)
    - [`op_scan()`](https://keras3.posit.co/reference/op_scan.md)
    - [`op_switch()`](https://keras3.posit.co/reference/op_switch.md)
    - [`op_dtype()`](https://keras3.posit.co/reference/op_dtype.md)
    - [`op_lstsq()`](https://keras3.posit.co/reference/op_lstsq.md)
    - [`op_image_hsv_to_rgb()`](https://keras3.posit.co/reference/op_image_hsv_to_rgb.md)
    - [`op_image_rgb_to_hsv()`](https://keras3.posit.co/reference/op_image_rgb_to_hsv.md)
  - Changes:
    - Added support for arbitrary, deeply nested input/output structures
      in Functional models (e.g. lists of lists of lists of inputs or
      outputs…)
    - Add support for `optional` Functional inputs.
      - [`keras_input()`](https://keras3.posit.co/reference/keras_input.md)
        gains an `optional` argument.
      - [`keras_model_sequential()`](https://keras3.posit.co/reference/keras_model_sequential.md)
        gains a `input_optional` argument.
    - Add support for `float8` inference for `Dense` and `EinsumDense`
      layers.
    - Enable
      [`layer_feature_space()`](https://keras3.posit.co/reference/layer_feature_space.md)
      to be used in a
      [tfdatasets](https://github.com/rstudio/tfdatasets) pipeline even
      when the backend isn’t TensorFlow.
    - [`layer_string_lookup()`](https://keras3.posit.co/reference/layer_string_lookup.md)
      can now take `tf$SparseTensor()` as input.
    - [`layer_string_lookup()`](https://keras3.posit.co/reference/layer_string_lookup.md)
      returns `"int64"` dtype by default in more modes now.
    - [`Layer()`](https://keras3.posit.co/reference/Layer.md) instances
      gain attributes `path` and `quantization_mode`.
    - `Metric()$variables` is now recursive.
    - Add `training` argument to `Model$compute_loss()`.
    - [`split_dataset()`](https://keras3.posit.co/reference/split_dataset.md)
      now supports nested structures in dataset.
    - All applications gain a `name` argument, accept a custom name.
    - [`layer_multi_head_attention()`](https://keras3.posit.co/reference/layer_multi_head_attention.md)
      gains a `seed` argument.
    - All losses gain a `dtype` argument.
    - [`loss_dice()`](https://keras3.posit.co/reference/loss_dice.md)
      gains an `axis` argument.
    - [`op_ctc_decode()`](https://keras3.posit.co/reference/op_ctc_decode.md),
      new default for `mask_index = 0`
    - All `op_image_*` functions now use default `data_format` value to
      [`config_image_data_format()`](https://keras3.posit.co/reference/config_image_data_format.md)
    - [`op_isclose()`](https://keras3.posit.co/reference/op_isclose.md)
      gains arguments `rtol`, `atol`, `equal_nan`.
    - [`save_model()`](https://keras3.posit.co/reference/save_model.md)
      gains argument `zipped`.
    - Bugs fixes and performance improvements.

## keras3 1.0.0

CRAN release: 2024-05-21

- Chains of `layer_*` calls with `|>` now instantiate layers in the same
  order as `%>%` pipe chains: left-hand-side first
  ([\#1440](https://github.com/rstudio/keras3/issues/1440)).

- [`iterate()`](https://rstudio.github.io/reticulate/reference/iterate.html),
  [`iter_next()`](https://rstudio.github.io/reticulate/reference/iterate.html)
  and
  [`as_iterator()`](https://rstudio.github.io/reticulate/reference/iterate.html)
  are now reexported from reticulate.

User facing changes with upstream Keras v3.3.3:

- new functions:
  [`op_slogdet()`](https://keras3.posit.co/reference/op_slogdet.md),
  [`op_psnr()`](https://keras3.posit.co/reference/op_psnr.md)

- [`clone_model()`](https://keras3.posit.co/reference/clone_model.md)
  gains new args: `call_function`, `recursive` Updated example usage.

- [`op_ctc_decode()`](https://keras3.posit.co/reference/op_ctc_decode.md)
  strategy argument has new default: `"greedy"`. Updated docs.

- [`loss_ctc()`](https://keras3.posit.co/reference/loss_ctc.md) default
  name fixed, changed to `"ctc"`

User facing changes with upstream Keras v3.3.2:

- new function:
  [`op_ctc_decode()`](https://keras3.posit.co/reference/op_ctc_decode.md)

- new function:
  [`op_eigh()`](https://keras3.posit.co/reference/op_eigh.md)

- new function:
  [`op_select()`](https://keras3.posit.co/reference/op_select.md)

- new function:
  [`op_vectorize()`](https://keras3.posit.co/reference/op_vectorize.md)

- new function:
  [`op_image_rgb_to_grayscale()`](https://keras3.posit.co/reference/op_image_rgb_to_grayscale.md)

- new function:
  [`loss_tversky()`](https://keras3.posit.co/reference/loss_tversky.md)

- new args: `layer_resizing(pad_to_aspect_ratio, fill_mode, fill_value)`

- new arg: `layer_embedding(weights)` for providing an initial weights
  matrix

- new args: `op_nan_to_num(nan, posinf, neginf)`

- new args:
  `op_image_resize(crop_to_aspect_ratio, pad_to_aspect_ratio, fill_mode, fill_value)`

- new args: `op_argmax(keepdims)` and `op_argmin(keepdims)`

- new arg: `clear_session(free_memory)` for clearing without invoking
  the garbage collector.

- [`metric_kl_divergence()`](https://keras3.posit.co/reference/metric_kl_divergence.md)
  and
  [`loss_kl_divergence()`](https://keras3.posit.co/reference/loss_kl_divergence.md)
  clip inputs (`y_true` and `y_pred`) to the `[0, 1]` range.

- new [`Layer()`](https://keras3.posit.co/reference/Layer.md)
  attributes: `metrics`, `dtype_policy`

- Added initial support for float8 training

- `layer_conv_*d()` layers now support LoRa

- [`op_digitize()`](https://keras3.posit.co/reference/op_digitize.md)
  now supports sparse tensors.

- Models and layers now return owned metrics recursively.

- Add pickling support for Keras models. (e.g., via
  [`reticulate::py_save_object()`](https://rstudio.github.io/reticulate/reference/py_save_object.html))
  Note that pickling is not recommended, prefer using Keras saving APIs.

## keras3 0.2.0

CRAN release: 2024-04-18

New functions:

- [`quantize_weights()`](https://keras3.posit.co/reference/quantize_weights.md):
  quantize model or layer weights in-place. Currently, only `Dense`,
  `EinsumDense`, and `Embedding` layers are supported (which is enough
  to cover the majority of transformers today)

- [`layer_mel_spectrogram()`](https://keras3.posit.co/reference/layer_mel_spectrogram.md)

- [`layer_flax_module_wrapper()`](https://keras3.posit.co/reference/layer_flax_module_wrapper.md)

- [`layer_jax_model_wrapper()`](https://keras3.posit.co/reference/layer_jax_model_wrapper.md)

- [`loss_dice()`](https://keras3.posit.co/reference/loss_dice.md)

- [`random_beta()`](https://keras3.posit.co/reference/random_beta.md)

- [`random_binomial()`](https://keras3.posit.co/reference/random_binomial.md)

- [`config_set_backend()`](https://keras3.posit.co/reference/config_set_backend.md):
  change the backend after Keras has initialized.

- [`config_dtype_policy()`](https://keras3.posit.co/reference/config_dtype_policy.md)

- [`config_set_dtype_policy()`](https://keras3.posit.co/reference/config_set_dtype_policy.md)

- New Ops

  - [`op_custom_gradient()`](https://keras3.posit.co/reference/op_custom_gradient.md)
  - [`op_batch_normalization()`](https://keras3.posit.co/reference/op_batch_normalization.md)
  - [`op_image_crop()`](https://keras3.posit.co/reference/op_image_crop.md)
  - [`op_divide_no_nan()`](https://keras3.posit.co/reference/op_divide_no_nan.md)
  - [`op_normalize()`](https://keras3.posit.co/reference/op_normalize.md)
  - [`op_correlate()`](https://keras3.posit.co/reference/op_correlate.md)
  - \`

- New family of linear algebra ops

  - [`op_cholesky()`](https://keras3.posit.co/reference/op_cholesky.md)
  - [`op_det()`](https://keras3.posit.co/reference/op_det.md)
  - [`op_eig()`](https://keras3.posit.co/reference/op_eig.md)
  - [`op_inv()`](https://keras3.posit.co/reference/op_inv.md)
  - [`op_lu_factor()`](https://keras3.posit.co/reference/op_lu_factor.md)
  - [`op_norm()`](https://keras3.posit.co/reference/op_norm.md)
  - [`op_erfinv()`](https://keras3.posit.co/reference/op_erfinv.md)
  - [`op_solve_triangular()`](https://keras3.posit.co/reference/op_solve_triangular.md)
  - [`op_svd()`](https://keras3.posit.co/reference/op_svd.md)

- [`audio_dataset_from_directory()`](https://keras3.posit.co/reference/audio_dataset_from_directory.md),
  [`image_dataset_from_directory()`](https://keras3.posit.co/reference/image_dataset_from_directory.md)
  and
  [`text_dataset_from_directory()`](https://keras3.posit.co/reference/text_dataset_from_directory.md)
  gain a `verbose` argument (default `TRUE`)

- [`image_dataset_from_directory()`](https://keras3.posit.co/reference/image_dataset_from_directory.md)
  gains `pad_to_aspect_ratio` argument (default `FALSE`)

- [`to_categorical()`](https://keras3.posit.co/reference/to_categorical.md),
  [`op_one_hot()`](https://keras3.posit.co/reference/op_one_hot.md), and
  [`fit()`](https://generics.r-lib.org/reference/fit.html) can now
  accept R factors, offset them to be 0-based (reported in `#1055`).

- [`op_convert_to_numpy()`](https://keras3.posit.co/reference/op_convert_to_numpy.md)
  now returns unconverted NumPy arrays.

- [`op_array()`](https://keras3.posit.co/reference/op_array.md) and
  [`op_convert_to_tensor()`](https://keras3.posit.co/reference/op_convert_to_tensor.md)
  no longer error when casting R doubles to integer types.

- [`export_savedmodel()`](https://rdrr.io/pkg/tensorflow/man/export_savedmodel.html)
  now works with a Jax backend.

- `Metric()$add_variable()` method gains arg: `aggregration`.

- `Layer()$add_weight()` method gains args: `autocast`, `regularizer`,
  `aggregation`.

- [`op_bincount()`](https://keras3.posit.co/reference/op_bincount.md),
  [`op_multi_hot()`](https://keras3.posit.co/reference/op_multi_hot.md),
  [`op_one_hot()`](https://keras3.posit.co/reference/op_one_hot.md), and
  [`layer_category_encoding()`](https://keras3.posit.co/reference/layer_category_encoding.md)
  now support sparse tensors.

- [`op_custom_gradient()`](https://keras3.posit.co/reference/op_custom_gradient.md)
  now supports the PyTorch backend

- [`layer_lstm()`](https://keras3.posit.co/reference/layer_lstm.md) and
  [`layer_gru()`](https://keras3.posit.co/reference/layer_gru.md) gain
  arg `use_cudnn`, default `'auto'`.

- Fixed an issue where
  [`application_preprocess_inputs()`](https://keras3.posit.co/reference/process_utils.md)
  would error if supplied an R array as input.

- Doc improvements.

## keras3 0.1.0

CRAN release: 2024-02-17

- The package has been rebuilt for Keras 3.0. Refer to
  <https://blogs.rstudio.com/ai/posts/2024-05-21-keras3/> for an
  overview and <https://keras3.posit.co> for the current up-to-date
  documentation.
