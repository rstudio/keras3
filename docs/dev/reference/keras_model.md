# Keras Model (Functional API)

A model is a directed acyclic graph of layers.

## Usage

``` r
keras_model(inputs = NULL, outputs = NULL, ...)
```

## Arguments

- inputs:

  Input tensor(s) (from
  [`keras_input()`](https://keras3.posit.co/dev/reference/keras_input.md))

- outputs:

  Output tensors (from calling layers with `inputs`)

- ...:

  Any additional arguments

## Value

A `Model` instance.

## Examples

    library(keras3)

    # input tensor
    inputs <- keras_input(shape = c(784))

    # outputs compose input + dense layers
    predictions <- inputs |>
      layer_dense(units = 64, activation = 'relu') |>
      layer_dense(units = 64, activation = 'relu') |>
      layer_dense(units = 10, activation = 'softmax')

    # create and compile model
    model <- keras_model(inputs = inputs, outputs = predictions)
    model |> compile(
      optimizer = 'rmsprop',
      loss = 'categorical_crossentropy',
      metrics = c('accuracy')
    )

## See also

Other model functions:  
[`get_config()`](https://keras3.posit.co/dev/reference/get_config.md)  
[`get_layer()`](https://keras3.posit.co/dev/reference/get_layer.md)  
[`get_state_tree()`](https://keras3.posit.co/dev/reference/get_state_tree.md)  
[`keras_model_sequential()`](https://keras3.posit.co/dev/reference/keras_model_sequential.md)  
[`pop_layer()`](https://keras3.posit.co/dev/reference/pop_layer.md)  
[`set_state_tree()`](https://keras3.posit.co/dev/reference/set_state_tree.md)  
[`summary.keras.src.models.model.Model()`](https://keras3.posit.co/dev/reference/summary.keras.src.models.model.Model.md)  

Other model creation:  
[`keras_input()`](https://keras3.posit.co/dev/reference/keras_input.md)  
[`keras_model_sequential()`](https://keras3.posit.co/dev/reference/keras_model_sequential.md)  
