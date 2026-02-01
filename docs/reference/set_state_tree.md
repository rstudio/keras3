# Assigns values to variables of the model.

This method takes a dictionary (named list) of nested variable values,
which represents the state tree of the model, and assigns them to the
corresponding variables of the model. The keys (list names) represent
the variable names (e.g., `'trainable_variables'`,
`'optimizer_variables'`), and the values are nested dictionaries
containing the variable paths and their corresponding values.

## Usage

``` r
set_state_tree(object, state_tree)
```

## Arguments

- object:

  A keras model.

- state_tree:

  A dictionary representing the state tree of the model. The keys are
  the variable names, and the values are nested dictionaries
  representing the variable paths and their values.

## See also

Other model functions:  
[`get_config()`](https://keras3.posit.co/reference/get_config.md)  
[`get_layer()`](https://keras3.posit.co/reference/get_layer.md)  
[`get_state_tree()`](https://keras3.posit.co/reference/get_state_tree.md)  
[`keras_model()`](https://keras3.posit.co/reference/keras_model.md)  
[`keras_model_sequential()`](https://keras3.posit.co/reference/keras_model_sequential.md)  
[`pop_layer()`](https://keras3.posit.co/reference/pop_layer.md)  
[`summary.keras.src.models.model.Model()`](https://keras3.posit.co/reference/summary.keras.src.models.model.Model.md)  
