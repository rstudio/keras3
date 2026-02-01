# Callback

`new_callback_class()` is an alias for
[`Callback()`](https://keras3.posit.co/reference/Callback.md). See
`?`[`Callback()`](https://keras3.posit.co/reference/Callback.md) for the
full documentation.

## Usage

``` r
new_callback_class(
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
