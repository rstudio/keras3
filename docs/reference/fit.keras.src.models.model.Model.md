# Train a model for a fixed number of epochs (dataset iterations).

Train a model for a fixed number of epochs (dataset iterations).

## Usage

``` r
# S3 method for class 'keras.src.models.model.Model'
fit(
  object,
  x = NULL,
  y = NULL,
  ...,
  batch_size = NULL,
  epochs = 1L,
  callbacks = NULL,
  validation_split = 0,
  validation_data = NULL,
  shuffle = TRUE,
  class_weight = NULL,
  sample_weight = NULL,
  initial_epoch = 1L,
  steps_per_epoch = NULL,
  validation_steps = NULL,
  validation_batch_size = NULL,
  validation_freq = 1L,
  verbose = getOption("keras.verbose", default = "auto"),
  view_metrics = getOption("keras.view_metrics", default = "auto")
)
```

## Arguments

- object:

  Keras model object

- x:

  Input data. It can be:

  - An array (or array-like), or a list of arrays (in case the model has
    multiple inputs).

  - A backend-native tensor, or a list of tensors (in case the model has
    multiple inputs).

  - A named list mapping input names to the corresponding array/tensors,
    if the model has named inputs.

  - A `tf.data.Dataset`. Should return a tuple of either
    `(inputs, targets)` or `(inputs, targets, sample_weights)`.

  - A generator returning `(inputs, targets)` or
    `(inputs, targets, sample_weights)`.

- y:

  Target data. Like the input data `x`, it can be either array(s) or
  backend-native tensor(s). If `x` is a TF Dataset or generator, `y`
  should not be specified (since targets will be obtained from `x`).

- ...:

  Additional arguments passed on to the model
  [`fit()`](https://generics.r-lib.org/reference/fit.html) method.

- batch_size:

  Integer or `NULL`. Number of samples per gradient update. If
  unspecified, `batch_size` will default to `32`. Do not specify the
  `batch_size` if your data is in the form of TF Datasets or generators,
  (since they generate batches).

- epochs:

  Integer. Number of epochs to train the model. An epoch is an iteration
  over the entire `x` and `y` data provided (unless the
  `steps_per_epoch` flag is set to something other than `NULL`). Note
  that in conjunction with `initial_epoch`, `epochs` is to be understood
  as "final epoch". The model is not trained for a number of iterations
  given by `epochs`, but merely until the epoch of index `epochs` is
  reached.

- callbacks:

  List of [`Callback()`](https://keras3.posit.co/reference/Callback.md)
  instances. List of callbacks to apply during training. See
  `callback_*`.

- validation_split:

  Fraction of the training data to be used as validation data. The model
  will set apart this fraction of the training data, will not train on
  it, and will evaluate the loss and any model metrics on this data at
  the end of each epoch. The validation data is selected from the last
  samples in the `x` and `y` data provided, before shuffling. This
  argument is only supported when `x` and `y` are made of arrays or
  tensors. If both `validation_data` and `validation_split` are
  provided, `validation_data` will override `validation_split`.

- validation_data:

  Data on which to evaluate the loss and any model metrics at the end of
  each epoch. The model will not be trained on this data. Thus, note the
  fact that the validation loss of data provided using
  `validation_split` or `validation_data` is not affected by
  regularization layers like noise and dropout. `validation_data` will
  override `validation_split`. It can be:

  - A tuple `(x_val, y_val)` of arrays or tensors.

  - A tuple `(x_val, y_val, val_sample_weights)` of arrays.

  - A generator returning `(inputs, targets)` or
    `(inputs, targets, sample_weights)`.

- shuffle:

  Boolean, whether to shuffle the training data before each epoch. This
  argument is ignored when `x` is a generator or a TF Dataset.

- class_weight:

  Optional named list mapping class indices (integers, 0-based) to a
  weight (float) value, used for weighting the loss function (during
  training only). This can be useful to tell the model to "pay more
  attention" to samples from an under-represented class. When
  `class_weight` is specified and targets have a rank of 2 or greater,
  either `y` must be one-hot encoded, or an explicit final dimension of
  `1` must be included for sparse class labels.

- sample_weight:

  Optional array or tensor of weights for the training samples, used for
  weighting the loss function (during training only). You can either
  pass a flat (1D) array or tensor with the same length as the input
  samples (1:1 mapping between weights and samples), or in the case of
  temporal data, you can pass a 2D array or tensor with shape
  `(samples, sequence_length)` to apply a different weight to every
  timestep of every sample. This argument is not supported when `x` is a
  `tf.data.Dataset`, or Python generator function. Instead, provide
  `sample_weights` as the third element of `x`. Note that sample
  weighting does not apply to metrics specified via the `metrics`
  argument in
  [`compile()`](https://generics.r-lib.org/reference/compile.html). To
  apply sample weighting to your metrics, you can specify them via the
  `weighted_metrics` in
  [`compile()`](https://generics.r-lib.org/reference/compile.html)
  instead.

- initial_epoch:

  Integer. Epoch at which to start training (useful for resuming a
  previous training run).

- steps_per_epoch:

  Integer or `NULL`. Total number of steps (batches of samples) before
  declaring one epoch finished and starting the next epoch. When
  training with input tensors or arrays, the default `NULL` means that
  the value used is the number of samples in your dataset divided by the
  batch size, or `1` if that cannot be determined. If `x` is a
  `tf.data.Dataset` or Python generator function, the epoch will run
  until the input dataset is exhausted. When passing an infinitely
  repeating dataset, you must specify the `steps_per_epoch` argument,
  otherwise the training will run indefinitely.

- validation_steps:

  Only relevant if `validation_data` is provided. Total number of steps
  (batches of samples) to draw before stopping when performing
  validation at the end of every epoch. If `validation_steps` is `NULL`,
  validation will run until the `validation_data` dataset is exhausted.
  In the case of an infinitely repeating dataset, it will run
  indefinitely. If `validation_steps` is specified and only part of the
  dataset is consumed, the evaluation will start from the beginning of
  the dataset at each epoch. This ensures that the same validation
  samples are used every time.

- validation_batch_size:

  Integer or `NULL`. Number of samples per validation batch. If
  unspecified, will default to `batch_size`. Do not specify the
  `validation_batch_size` if your data is in the form of TF Datasets or
  generator instances (since they generate batches).

- validation_freq:

  Only relevant if validation data is provided. Specifies how many
  training epochs to run before a new validation run is performed, e.g.
  `validation_freq=2` runs validation every 2 epochs.

- verbose:

  `"auto"`, `0`, `1`, or `2`. Verbosity mode. `0` = silent, `1` =
  progress bar, `2` = one line per epoch. `"auto"` becomes 1 for most
  cases, `2` if in a knitr render or running on a distributed training
  server. Note that the progress bar is not particularly useful when
  logged to a file, so `verbose=2` is recommended when not running
  interactively (e.g., in a production environment). Defaults to
  `"auto"`.

- view_metrics:

  View realtime plot of training metrics (by epoch). The default
  (`"auto"`) will display the plot when running within RStudio,
  `metrics` were specified during model
  [`compile()`](https://generics.r-lib.org/reference/compile.html),
  `epochs > 1` and `verbose > 0`. Set the global
  `options(keras.view_metrics = )` option to establish a different
  default.

## Value

A `keras_training_history` object, which is a named list:
`list(params = <params>, metrics = <metrics>")`, with S3 methods
[`print()`](https://rdrr.io/r/base/print.html),
[`plot()`](https://rdrr.io/r/graphics/plot.default.html), and
[`as.data.frame()`](https://rdrr.io/r/base/as.data.frame.html). The
metrics field is a record of training loss values and metrics values at
successive epochs, as well as validation loss values and validation
metrics values (if applicable).

## Details

Unpacking behavior for iterator-like inputs:

A common pattern is to pass an iterator like object such as a
`tf.data.Dataset` or a generator to
[`fit()`](https://generics.r-lib.org/reference/fit.html), which will in
fact yield not only features (`x`) but optionally targets (`y`) and
sample weights (`sample_weight`). Keras requires that the output of such
iterator-likes be unambiguous. The iterator should return a
[`tuple()`](https://rstudio.github.io/reticulate/reference/tuple.html)
of length 1, 2, or 3, where the optional second and third elements will
be used for `y` and `sample_weight` respectively. Any other type
provided will be wrapped in a length-one
[`tuple()`](https://rstudio.github.io/reticulate/reference/tuple.html),
effectively treating everything as `x`. When yielding named lists, they
should still adhere to the top-level tuple structure, e.g.
`tuple(list(x0 = x0, x = x1), y)`. Keras will not attempt to separate
features, targets, and weights from the keys of a single dict.

## See also

- <https://keras.io/api/models/model_training_apis#fit-method>
