# Runs a single gradient update on a single batch of data.

Runs a single gradient update on a single batch of data.

## Usage

``` r
train_on_batch(object, x, y = NULL, sample_weight = NULL, class_weight = NULL)
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

- class_weight:

  Optional named list mapping class indices (integers, 0-based) to a
  weight (float) to apply to the model's loss for the samples from this
  class during training. This can be useful to tell the model to "pay
  more attention" to samples from an under-represented class. When
  `class_weight` is specified and targets have a rank of 2 or greater,
  either `y` must be one-hot encoded, or an explicit final dimension of
  1 must be included for sparse class labels.

## Value

A scalar loss value (when no metrics), or a named list of loss and
metric values (if there are metrics). The property `model$metrics_names`
will give you the display labels for the scalar outputs.

## See also

- <https://keras.io/api/models/model_training_apis#trainonbatch-method>

Other model training:  
[`compile.keras.src.models.model.Model()`](https://keras3.posit.co/reference/compile.keras.src.models.model.Model.md)  
[`evaluate.keras.src.models.model.Model()`](https://keras3.posit.co/reference/evaluate.keras.src.models.model.Model.md)  
[`predict.keras.src.models.model.Model()`](https://keras3.posit.co/reference/predict.keras.src.models.model.Model.md)  
[`predict_on_batch()`](https://keras3.posit.co/reference/predict_on_batch.md)  
[`test_on_batch()`](https://keras3.posit.co/reference/test_on_batch.md)  
