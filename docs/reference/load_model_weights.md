# Load the weights from a single file or sharded files.

Weights are loaded based on the network's topology. This means the
architecture should be the same as when the weights were saved. Note
that layers that don't have weights are not taken into account in the
topological ordering, so adding or removing layers is fine as long as
they don't have weights.

**Sharding**

When loading sharded weights, specify a `filepath` ending in
`".weights.json"` (the configuration file), with the corresponding shard
files (`*_xxxxx.weights.h5`) located alongside it.

**Partial weight loading**

If you have modified your model, for instance by adding a new layer
(with weights) or by changing the shape of the weights of a layer, you
can choose to ignore errors and continue loading by setting
`skip_mismatch=TRUE`. In this case any layer with mismatching weights
will be skipped. A warning will be displayed for each skipped layer.

## Usage

``` r
load_model_weights(model, filepath, skip_mismatch = FALSE, ...)
```

## Arguments

- model:

  A keras model.

- filepath:

  Path or path-like object to the weights. Accepts `.weights.h5`, legacy
  `.h5`, or sharded weights through a `.weights.json` manifest sitting
  alongside the shard files (`*_xxxxx.weights.h5`).

- skip_mismatch:

  Boolean, whether to skip loading of layers where there is a mismatch
  in the number of weights, or a mismatch in the shape of the weights.

- ...:

  For forward/backward compatability.

## Value

This is called primarily for side effects. `model` is returned,
invisibly, to enable usage with the pipe.

## Examples

    model |> load_model_weights("model.weights.h5")
    model |> load_model_weights("model.weights.json")

## See also

- <https://keras.io/api/models/model_saving_apis/weights_saving_and_loading#loadweights-method>

Other saving and loading functions:  
[`export_savedmodel.keras.src.models.model.Model()`](https://keras3.posit.co/reference/export_savedmodel.keras.src.models.model.Model.md)  
[`layer_tfsm()`](https://keras3.posit.co/reference/layer_tfsm.md)  
[`load_model()`](https://keras3.posit.co/reference/load_model.md)  
[`register_keras_serializable()`](https://keras3.posit.co/reference/register_keras_serializable.md)  
[`save_model()`](https://keras3.posit.co/reference/save_model.md)  
[`save_model_config()`](https://keras3.posit.co/reference/save_model_config.md)  
[`save_model_weights()`](https://keras3.posit.co/reference/save_model_weights.md)  
[`with_custom_object_scope()`](https://keras3.posit.co/reference/with_custom_object_scope.md)  
