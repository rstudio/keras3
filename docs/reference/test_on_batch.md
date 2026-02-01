# Test the model on a single batch of samples.

Test the model on a single batch of samples.

## Usage

``` r
test_on_batch(object, x, y = NULL, sample_weight = NULL, ...)
```

## Arguments

- object:

  Keras model object

- x:

  Input data. Must be array-like.

- y:

  Target data. Must be array-like.

- sample_weight:

  Optional array of the same length as x, containing weights to apply to
  the model's loss for each sample. In the case of temporal data, you
  can pass a 2D array with shape `(samples, sequence_length)`, to apply
  a different weight to every timestep of every sample.

- ...:

  for forward/backward compatability

## Value

A scalar loss value (when no metrics), or a named list of loss and
metric values (if there are metrics).

## See also

- <https://keras.io/api/models/model_training_apis#testonbatch-method>

Other model training:  
[`compile.keras.src.models.model.Model()`](https://keras3.posit.co/reference/compile.keras.src.models.model.Model.md)  
[`evaluate.keras.src.models.model.Model()`](https://keras3.posit.co/reference/evaluate.keras.src.models.model.Model.md)  
[`predict.keras.src.models.model.Model()`](https://keras3.posit.co/reference/predict.keras.src.models.model.Model.md)  
[`predict_on_batch()`](https://keras3.posit.co/reference/predict_on_batch.md)  
[`train_on_batch()`](https://keras3.posit.co/reference/train_on_batch.md)  
