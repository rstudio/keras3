# Retrieves a layer based on either its name (unique) or index.

Indices are based on order of horizontal graph traversal (bottom-up) and
are 1-based. If `name` and `index` are both provided, `index` will take
precedence.

## Usage

``` r
get_layer(object, name = NULL, index = NULL)
```

## Arguments

- object:

  Keras model object

- name:

  String, name of layer.

- index:

  Integer, index of layer (1-based). Also valid are negative values,
  which count from the end of model.

## Value

A layer instance.

## See also

Other model functions:  
[`get_config()`](https://keras3.posit.co/dev/reference/get_config.md)  
[`get_state_tree()`](https://keras3.posit.co/dev/reference/get_state_tree.md)  
[`keras_model()`](https://keras3.posit.co/dev/reference/keras_model.md)  
[`keras_model_sequential()`](https://keras3.posit.co/dev/reference/keras_model_sequential.md)  
[`pop_layer()`](https://keras3.posit.co/dev/reference/pop_layer.md)  
[`set_state_tree()`](https://keras3.posit.co/dev/reference/set_state_tree.md)  
[`summary.keras.src.models.model.Model()`](https://keras3.posit.co/dev/reference/summary.keras.src.models.model.Model.md)  
