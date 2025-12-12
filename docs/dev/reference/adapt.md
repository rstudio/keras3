# Fits the state of the preprocessing layer to the data being passed

Fits the state of the preprocessing layer to the data being passed

## Usage

``` r
adapt(object, data, ..., batch_size = NULL, steps = NULL)
```

## Arguments

- object:

  Preprocessing layer object

- data:

  The data to train on. It can be passed either as a `tf.data.Dataset`
  or as an R array.

- ...:

  Used for forwards and backwards compatibility. Passed on to the
  underlying method.

- batch_size:

  Integer or `NULL`. Number of asamples per state update. If
  unspecified, `batch_size` will default to `32`. Do not specify the
  batch_size if your data is in the form of a TF Dataset or a generator
  (since they generate batches).

- steps:

  Integer or `NULL`. Total number of steps (batches of samples) When
  training with input tensors such as TensorFlow data tensors, the
  default `NULL` is equal to the number of samples in your dataset
  divided by the batch size, or `1` if that cannot be determined. If x
  is a `tf.data.Dataset`, and `steps` is `NULL`, the epoch will run
  until the input dataset is exhausted. When passing an infinitely
  repeating dataset, you must specify the steps argument. This argument
  is not supported with array inputs.

## Value

Returns `object`, invisibly.

## Details

After calling `adapt` on a layer, a preprocessing layer's state will not
update during training. In order to make preprocessing layers efficient
in any distribution context, they are kept constant with respect to any
compiled `tf.Graph`s that call the layer. This does not affect the layer
use when adapting each layer only once, but if you adapt a layer
multiple times you will need to take care to re-compile any compiled
functions as follows:

- If you are adding a preprocessing layer to a keras model, you need to
  call `compile(model)` after each subsequent call to `adapt()`.

- If you are calling a preprocessing layer inside
  [`tfdatasets::dataset_map()`](https://rdrr.io/pkg/tfdatasets/man/dataset_map.html),
  you should call `dataset_map()` again on the input `Dataset` after
  each `adapt()`.

- If you are using a
  [`tensorflow::tf_function()`](https://rdrr.io/pkg/tensorflow/man/tf_function.html)
  directly which calls a preprocessing layer, you need to call
  `tf_function()` again on your callable after each subsequent call to
  `adapt()`.

[`keras_model()`](https://keras3.posit.co/dev/reference/keras_model.md)
example with multiple adapts:

    layer <- layer_normalization(axis = NULL)
    adapt(layer, c(0, 2))
    model <- keras_model_sequential() |> layer()
    predict(model, c(0, 1, 2), verbose = FALSE) # [1] -1  0  1

    ## [1] -1  0  1

    adapt(layer, c(-1, 1))
    compile(model)  # This is needed to re-compile model.predict!
    predict(model, c(0, 1, 2), verbose = FALSE) # [1] 0 1 2

    ## [1] 0 1 2

`tfdatasets` example with multiple adapts:

    layer <- layer_normalization(axis = NULL)
    adapt(layer, c(0, 2))
    input_ds <- tfdatasets::range_dataset(0, 3)
    normalized_ds <- input_ds |>
      tfdatasets::dataset_map(layer)
    str(tfdatasets::iterate(normalized_ds))

    ## List of 3
    ##  $ :<tf.Tensor: shape=(1), dtype=float32, numpy=array([-1.], dtype=float32)>
    ##  $ :<tf.Tensor: shape=(1), dtype=float32, numpy=array([0.], dtype=float32)>
    ##  $ :<tf.Tensor: shape=(1), dtype=float32, numpy=array([1.], dtype=float32)>

    adapt(layer, c(-1, 1))
    normalized_ds <- input_ds |>
      tfdatasets::dataset_map(layer) # Re-map over the input dataset.

    normalized_ds |>
      tfdatasets::as_array_iterator() |>
      tfdatasets::iterate(simplify = FALSE) |>
      str()

    ## List of 3
    ##  $ : num [1(1d)] 0
    ##  $ : num [1(1d)] 1
    ##  $ : num [1(1d)] 2
