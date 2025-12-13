# Layer/Model configuration

A layer config is an object returned from `get_config()` that contains
the configuration of a layer or model. The same layer or model can be
reinstantiated later (without its trained weights) from this
configuration using `from_config()`. The config does not include
connectivity information, nor the class name (those are handled
externally).

## Usage

``` r
get_config(object)

from_config(config, custom_objects = NULL)
```

## Arguments

- object:

  Layer or model object

- config:

  Object with layer or model configuration

- custom_objects:

  list of custom objects needed to instantiate the layer, e.g., custom
  layers defined by
  [`new_layer_class()`](https://keras3.posit.co/dev/reference/new_layer_class.md)
  or similar.

## Value

`get_config()` returns an object with the configuration, `from_config()`
returns a re-instantiation of the object.

## Note

Objects returned from `get_config()` are not serializable via RDS. If
you want to save and restore a model across sessions, you can use
[`save_model_config()`](https://keras3.posit.co/dev/reference/save_model_config.md)
(for model configuration only, not weights) or
[`save_model()`](https://keras3.posit.co/dev/reference/save_model.md) to
save the model configuration and weights to the filesystem.

## See also

Other model functions:  
[`get_layer()`](https://keras3.posit.co/dev/reference/get_layer.md)  
[`get_state_tree()`](https://keras3.posit.co/dev/reference/get_state_tree.md)  
[`keras_model()`](https://keras3.posit.co/dev/reference/keras_model.md)  
[`keras_model_sequential()`](https://keras3.posit.co/dev/reference/keras_model_sequential.md)  
[`pop_layer()`](https://keras3.posit.co/dev/reference/pop_layer.md)  
[`set_state_tree()`](https://keras3.posit.co/dev/reference/set_state_tree.md)  
[`summary.keras.src.models.model.Model()`](https://keras3.posit.co/dev/reference/summary.keras.src.models.model.Model.md)  

Other layer methods:  
[`count_params()`](https://keras3.posit.co/dev/reference/count_params.md)  
[`get_weights()`](https://keras3.posit.co/dev/reference/get_weights.md)  
[`quantize_weights()`](https://keras3.posit.co/dev/reference/quantize_weights.md)  
[`reset_state()`](https://keras3.posit.co/dev/reference/reset_state.md)  
