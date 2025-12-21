# Remove the last layer in a Sequential model

Remove the last layer in a Sequential model

## Usage

``` r
pop_layer(object, rebuild = TRUE)
```

## Arguments

- object:

  Sequential keras model object

- rebuild:

  `bool`. Whether to rebuild the model after removing the layer.
  Defaults to `TRUE`.

## Value

The removed layer.

## See also

Other model functions:  
[`get_config()`](https://keras3.posit.co/reference/get_config.md)  
[`get_layer()`](https://keras3.posit.co/reference/get_layer.md)  
[`get_state_tree()`](https://keras3.posit.co/reference/get_state_tree.md)  
[`keras_model()`](https://keras3.posit.co/reference/keras_model.md)  
[`keras_model_sequential()`](https://keras3.posit.co/reference/keras_model_sequential.md)  
[`set_state_tree()`](https://keras3.posit.co/reference/set_state_tree.md)  
[`summary.keras.src.models.model.Model()`](https://keras3.posit.co/reference/summary.keras.src.models.model.Model.md)  
