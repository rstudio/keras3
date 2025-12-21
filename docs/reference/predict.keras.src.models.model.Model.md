# Generates output predictions for the input samples.

Generates output predictions for the input samples.

## Usage

``` r
# S3 method for class 'keras.src.models.model.Model'
predict(
  object,
  x,
  ...,
  batch_size = NULL,
  verbose = getOption("keras.verbose", default = "auto"),
  steps = NULL,
  callbacks = NULL
)
```

## Arguments

- object:

  Keras model object

- x:

  Input samples. It can be:

  - A array (or array-like), or a list of arrays (in case the model has
    multiple inputs).

  - A backend-native tensor, or a list of tensors (in case the model has
    multiple inputs).

  - A TF Dataset.

  - A Python generator function.

- ...:

  For forward/backward compatability.

- batch_size:

  Integer or `NULL`. Number of samples per batch of computation. If
  unspecified, `batch_size` will default to `32`. Do not specify the
  `batch_size` if your input data `x` is in the form of a TF Dataset or
  a generator (since they generate batches).

- verbose:

  `"auto"`, `0`, `1`, or `2`. Verbosity mode. `0` = silent, `1` =
  progress bar, `2` = one line per epoch. `"auto"` becomes 1 for most
  cases, `2` if in a knitr render or running on a distributed training
  server. Note that the progress bar is not particularly useful when
  logged to a file, so `verbose=2` is recommended when not running
  interactively (e.g., in a production environment). Defaults to
  `"auto"`.

- steps:

  Total number of steps (batches of samples) to draw before declaring
  the prediction round finished. If `steps` is `NULL`,
  [`predict()`](https://rdrr.io/r/stats/predict.html) will run until `x`
  is exhausted. In the case of an infinitely repeating dataset,
  [`predict()`](https://rdrr.io/r/stats/predict.html) will run
  indefinitely.

- callbacks:

  List of `Callback` instances. List of callbacks to apply during
  prediction.

## Value

R array(s) of predictions.

## Details

Computation is done in batches. This method is designed for batch
processing of large numbers of inputs. It is not intended for use inside
of loops that iterate over your data and process small numbers of inputs
at a time.

For small numbers of inputs that fit in one batch, directly call the
model `model$call` for faster execution, e.g., `model(x)`, or
`model(x, training = FALSE)` if you have layers such as
`BatchNormalization` that behave differently during inference.

## Note

See [this FAQ
entry](https://keras.io/getting_started/faq/#whats-the-difference-between-model-methods-predict-and-call)
for more details about the difference between `Model` methods
[`predict()`](https://rdrr.io/r/stats/predict.html) and
[`call()`](https://rdrr.io/r/base/call.html).

## See also

- <https://keras.io/api/models/model_training_apis#predict-method>

Other model training:  
[`compile.keras.src.models.model.Model()`](https://keras3.posit.co/reference/compile.keras.src.models.model.Model.md)  
[`evaluate.keras.src.models.model.Model()`](https://keras3.posit.co/reference/evaluate.keras.src.models.model.Model.md)  
[`predict_on_batch()`](https://keras3.posit.co/reference/predict_on_batch.md)  
[`test_on_batch()`](https://keras3.posit.co/reference/test_on_batch.md)  
[`train_on_batch()`](https://keras3.posit.co/reference/train_on_batch.md)  
