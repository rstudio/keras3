# Save and load model configuration as JSON

Save and re-load models configurations as JSON. Note that the
representation does not include the weights, only the architecture.

## Usage

``` r
save_model_config(model, filepath = NULL, overwrite = FALSE)

load_model_config(filepath, custom_objects = NULL)
```

## Arguments

- model:

  Model object to save

- filepath:

  path to json file with the model config.

- overwrite:

  Whether we should overwrite any existing model configuration json at
  `filepath`, or instead ask the user via an interactive prompt.

- custom_objects:

  Optional named list mapping names to custom classes or functions to be
  considered during deserialization.

## Value

This is called primarily for side effects. `model` is returned,
invisibly, to enable usage with the pipe.

## Details

Note: `save_model_config()` serializes the model to JSON using
[`serialize_keras_object()`](https://keras3.posit.co/reference/serialize_keras_object.md),
not [`get_config()`](https://keras3.posit.co/reference/get_config.md).
[`serialize_keras_object()`](https://keras3.posit.co/reference/serialize_keras_object.md)
returns a superset of
[`get_config()`](https://keras3.posit.co/reference/get_config.md), with
additional information needed to create the class object needed to
restore the model. See example for how to extract the
[`get_config()`](https://keras3.posit.co/reference/get_config.md) value
from a saved model.

## Example

    model <- keras_model_sequential(input_shape = 10) |> layer_dense(10)
    file <- tempfile("model-config-", fileext = ".json")
    save_model_config(model, file)

    # load a new model instance with the same architecture but different weights
    model2 <- load_model_config(file)

    stopifnot(exprs = {
      all.equal(get_config(model), get_config(model2))

      # To extract the `get_config()` value from a saved model config:
      all.equal(
          get_config(model),
          structure(jsonlite::read_json(file)$config,
                    "__class__" = keras_model_sequential()$`__class__`)
      )
    })

## See also

Other saving and loading functions:  
[`export_savedmodel.keras.src.models.model.Model()`](https://keras3.posit.co/reference/export_savedmodel.keras.src.models.model.Model.md)  
[`layer_tfsm()`](https://keras3.posit.co/reference/layer_tfsm.md)  
[`load_model()`](https://keras3.posit.co/reference/load_model.md)  
[`load_model_weights()`](https://keras3.posit.co/reference/load_model_weights.md)  
[`register_keras_serializable()`](https://keras3.posit.co/reference/register_keras_serializable.md)  
[`save_model()`](https://keras3.posit.co/reference/save_model.md)  
[`save_model_weights()`](https://keras3.posit.co/reference/save_model_weights.md)  
[`with_custom_object_scope()`](https://keras3.posit.co/reference/with_custom_object_scope.md)  
