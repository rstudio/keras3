# Define a custom `Callback` class

Callbacks can be passed to keras methods such as
[`fit()`](https://generics.r-lib.org/reference/fit.html),
[`evaluate()`](https://rdrr.io/pkg/tensorflow/man/evaluate.html), and
[`predict()`](https://rdrr.io/r/stats/predict.html) in order to hook
into the various stages of the model training, evaluation, and inference
lifecycle.

To create a custom callback, call `Callback()` and override the method
associated with the stage of interest.

## Usage

``` r
Callback(
  classname,
  on_epoch_begin = NULL,
  on_epoch_end = NULL,
  on_train_begin = NULL,
  on_train_end = NULL,
  on_train_batch_begin = NULL,
  on_train_batch_end = NULL,
  on_test_begin = NULL,
  on_test_end = NULL,
  on_test_batch_begin = NULL,
  on_test_batch_end = NULL,
  on_predict_begin = NULL,
  on_predict_end = NULL,
  on_predict_batch_begin = NULL,
  on_predict_batch_end = NULL,
  ...,
  public = list(),
  private = list(),
  inherit = NULL,
  parent_env = parent.frame()
)
```

## Arguments

- classname:

  String, the name of the custom class. (Conventionally, CamelCase).

- on_epoch_begin:

      \(epoch, logs = NULL)

  Called at the start of an epoch.

  Subclasses should override for any actions to run. This function
  should only be called during TRAIN mode.

  Args:

  - `epoch`: Integer, index of epoch.

  - `logs`: Named List. Currently no data is passed to this argument for
    this method but that may change in the future.

- on_epoch_end:

      \(epoch, logs = NULL)

  Called at the end of an epoch.

  Subclasses should override for any actions to run. This function
  should only be called during TRAIN mode.

  Args:

  - `epoch`: Integer, index of epoch.

  - `logs`: Named List, metric results for this training epoch, and for
    the validation epoch if validation is performed. Validation result
    keys are prefixed with `val_`. For training epoch, the values of the
    `Model`'s metrics are returned. Example:
    `list(loss = 0.2, accuracy = 0.7)`.

- on_train_begin:

      \(logs = NULL)

  Called at the beginning of training.

  Subclasses should override for any actions to run.

  Args:

  - `logs`: Named list. Currently no data is passed to this argument for
    this method but that may change in the future.

- on_train_end:

      \(logs = NULL)

  Called at the end of training.

  Subclasses should override for any actions to run.

  Args:

  - `logs`: Named list. Currently the output of the last call to
    `on_epoch_end()` is passed to this argument for this method but that
    may change in the future.

- on_train_batch_begin:

      \(batch, logs = NULL)

  Called at the beginning of a training batch in
  [`fit()`](https://generics.r-lib.org/reference/fit.html) methods.

  Subclasses should override for any actions to run.

  Note that if the `steps_per_execution` argument to `compile` in
  `Model` is set to `N`, this method will only be called every `N`
  batches.

  Args:

  - `batch`: Integer, index of batch within the current epoch.

  - `logs`: Named list. Currently no data is passed to this argument for
    this method but that may change in the future.

- on_train_batch_end:

      \(batch, logs=NULL)

  Called at the end of a training batch in
  [`fit()`](https://generics.r-lib.org/reference/fit.html) methods.

  Subclasses should override for any actions to run.

  Note that if the `steps_per_execution` argument to `compile` in
  `Model` is set to `N`, this method will only be called every `N`
  batches.

  Args:

  - `batch`: Integer, index of batch within the current epoch.

  - `logs`: Named list. Aggregated metric results up until this batch.

- on_test_begin:

      \(logs = NULL)

  Called at the beginning of evaluation or validation.

  Subclasses should override for any actions to run.

  Args:

  - `logs`: Named list. Currently no data is passed to this argument for
    this method but that may change in the future.

- on_test_end:

      \(logs = NULL)

  Called at the end of evaluation or validation.

  Subclasses should override for any actions to run.

  Args:

  - `logs`: Named list. Currently the output of the last call to
    `on_test_batch_end()` is passed to this argument for this method but
    that may change in the future.

- on_test_batch_begin:

      \(batch, logs = NULL)

  Called at the beginning of a batch in
  [`evaluate()`](https://rdrr.io/pkg/tensorflow/man/evaluate.html)
  methods.

  Also called at the beginning of a validation batch in the
  [`fit()`](https://generics.r-lib.org/reference/fit.html) methods, if
  validation data is provided.

  Subclasses should override for any actions to run.

  Note that if the `steps_per_execution` argument to
  [`compile()`](https://generics.r-lib.org/reference/compile.html) in
  `Model` is set to `N`, this method will only be called every `N`
  batches.

  Args:

  - `batch`: Integer, index of batch within the current epoch.

  - `logs`: Named list. Currently no data is passed to this argument for
    this method but that may change in the future.

- on_test_batch_end:

      \(batch, logs = NULL)

  Called at the end of a batch in
  [`evaluate()`](https://rdrr.io/pkg/tensorflow/man/evaluate.html)
  methods.

  Also called at the end of a validation batch in the
  [`fit()`](https://generics.r-lib.org/reference/fit.html) methods, if
  validation data is provided.

  Subclasses should override for any actions to run.

  Note that if the `steps_per_execution` argument to
  [`compile()`](https://generics.r-lib.org/reference/compile.html) in
  `Model` is set to `N`, this method will only be called every `N`
  batches.

  Args:

  - `batch`: Integer, index of batch within the current epoch.

  - `logs`: Named list. Aggregated metric results up until this batch.

- on_predict_begin:

      \(logs = NULL)

  Called at the beginning of prediction.

  Subclasses should override for any actions to run.

  Args:

  - `logs`: Named list. Currently no data is passed to this argument for
    this method but that may change in the future.

- on_predict_end:

      \(logs = NULL)

  Called at the end of prediction.

  Subclasses should override for any actions to run.

  Args:

  - `logs`: Named list. Currently no data is passed to this argument for
    this method but that may change in the future.

- on_predict_batch_begin:

      \(batch, logs = NULL)

  Called at the beginning of a batch in
  [`predict()`](https://rdrr.io/r/stats/predict.html) methods.

  Subclasses should override for any actions to run.

  Note that if the `steps_per_execution` argument to
  [`compile()`](https://generics.r-lib.org/reference/compile.html) in
  `Model` is set to `N`, this method will only be called every `N`
  batches.

  Args:

  - `batch`: Integer, index of batch within the current epoch.

  - `logs`: Named list. Currently no data is passed to this argument for
    this method but that may change in the future.

- on_predict_batch_end:

      \(batch, logs = NULL)

  Called at the end of a batch in
  [`predict()`](https://rdrr.io/r/stats/predict.html) methods.

  Subclasses should override for any actions to run.

  Note that if the `steps_per_execution` argument to `compile` in
  `Model` is set to `N`, this method will only be called every `N`
  batches.

  Args:

  - `batch`: Integer, index of batch within the current epoch.

  - `logs`: Named list. Aggregated metric results up until this batch.

- ..., public:

  Additional methods or public members of the custom class.

- private:

  Named list of R objects (typically, functions) to include in instance
  private environments. `private` methods will have all the same symbols
  in scope as public methods (See section "Symbols in Scope"). Each
  instance will have it's own `private` environment. Any objects in
  `private` will be invisible from the Keras framework and the Python
  runtime.

- inherit:

  What the custom class will subclass. By default, the base keras class.

- parent_env:

  The R environment that all class methods will have as a grandparent.

## Value

A function that returns the custom `Callback` instances, similar to the
builtin callback functions.

## Examples

    training_finished <- FALSE
    callback_mark_finished <- Callback("MarkFinished",
      on_train_end = function(logs = NULL) {
        training_finished <<- TRUE
      }
    )

    model <- keras_model_sequential(input_shape = c(1)) |>
      layer_dense(1)
    model |> compile(loss = 'mean_squared_error')
    model |> fit(op_ones(c(1, 1)), op_ones(c(1, 1)),
                 callbacks = callback_mark_finished())
    stopifnot(isTRUE(training_finished))

All R function custom methods (public and private) will have the
following symbols in scope:

- `self`: the `Layer` instance.

- `super`: the `Layer` superclass.

- `private`: An R environment specific to the class instance. Any
  objects defined here will be invisible to the Keras framework.

- `__class__` the current class type object. This will also be available
  as an alias symbol, the value supplied to `Layer(classname = )`

## Attributes (accessible via `self$`)

- `params`: Named list, Training parameters (e.g. verbosity, batch size,
  number of epochs, ...).

- `model`: Instance of `Model`. Reference of the model being trained.

The `logs` named list that callback methods take as argument will
contain keys for quantities relevant to the current batch or epoch (see
method-specific docstrings).

## Symbols in scope

All R function custom methods (public and private) will have the
following symbols in scope:

- `self`: The custom class instance.

- `super`: The custom class superclass.

- `private`: An R environment specific to the class instance. Any
  objects assigned here are invisible to the Keras framework.

- `__class__` and `as.symbol(classname)`: the custom class type object.

## See also

- <https://keras.io/api/callbacks/base_callback#callback-class>

Other callbacks:  
[`callback_backup_and_restore()`](https://keras3.posit.co/reference/callback_backup_and_restore.md)  
[`callback_csv_logger()`](https://keras3.posit.co/reference/callback_csv_logger.md)  
[`callback_early_stopping()`](https://keras3.posit.co/reference/callback_early_stopping.md)  
[`callback_lambda()`](https://keras3.posit.co/reference/callback_lambda.md)  
[`callback_learning_rate_scheduler()`](https://keras3.posit.co/reference/callback_learning_rate_scheduler.md)  
[`callback_model_checkpoint()`](https://keras3.posit.co/reference/callback_model_checkpoint.md)  
[`callback_reduce_lr_on_plateau()`](https://keras3.posit.co/reference/callback_reduce_lr_on_plateau.md)  
[`callback_remote_monitor()`](https://keras3.posit.co/reference/callback_remote_monitor.md)  
[`callback_swap_ema_weights()`](https://keras3.posit.co/reference/callback_swap_ema_weights.md)  
[`callback_tensorboard()`](https://keras3.posit.co/reference/callback_tensorboard.md)  
[`callback_terminate_on_nan()`](https://keras3.posit.co/reference/callback_terminate_on_nan.md)  
