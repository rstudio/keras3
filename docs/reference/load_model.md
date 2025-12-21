# Loads a model saved via `save_model()`.

Loads a model saved via
[`save_model()`](https://keras3.posit.co/reference/save_model.md).

## Usage

``` r
load_model(model, custom_objects = NULL, compile = TRUE, safe_mode = TRUE)
```

## Arguments

- model:

  string, path to the saved model file, or a raw vector, as returned by
  `save_model(filepath = NULL)`

- custom_objects:

  Optional named list mapping names to custom classes or functions to be
  considered during deserialization.

- compile:

  Boolean, whether to compile the model after loading.

- safe_mode:

  Boolean, whether to disallow unsafe `lambda` deserialization. When
  `safe_mode=FALSE`, loading an object has the potential to trigger
  arbitrary code execution. This argument is only applicable to the
  Keras v3 model format. Defaults to `TRUE`.

## Value

A Keras model instance. If the original model was compiled, and the
argument `compile = TRUE` is set, then the returned model will be
compiled. Otherwise, the model will be left uncompiled.

## Examples

    model <- keras_model_sequential(input_shape = c(3)) |>
      layer_dense(5) |>
      layer_activation_softmax()

    model |> save_model("model.keras")
    loaded_model <- load_model("model.keras")

    x <- random_uniform(c(10, 3))
    stopifnot(all.equal(
      model |> predict(x),
      loaded_model |> predict(x)
    ))

Note that the model variables may have different name values (`var$name`
property, e.g. `"dense_1/kernel:0"`) after being reloaded. It is
recommended that you use layer attributes to access specific variables,
e.g. `model |> get_layer("dense_1") |> _$kernel`.

## See also

- <https://keras.io/api/models/model_saving_apis/model_saving_and_loading#loadmodel-function>

Other saving and loading functions:  
[`export_savedmodel.keras.src.models.model.Model()`](https://keras3.posit.co/reference/export_savedmodel.keras.src.models.model.Model.md)  
[`layer_tfsm()`](https://keras3.posit.co/reference/layer_tfsm.md)  
[`load_model_weights()`](https://keras3.posit.co/reference/load_model_weights.md)  
[`register_keras_serializable()`](https://keras3.posit.co/reference/register_keras_serializable.md)  
[`save_model()`](https://keras3.posit.co/reference/save_model.md)  
[`save_model_config()`](https://keras3.posit.co/reference/save_model_config.md)  
[`save_model_weights()`](https://keras3.posit.co/reference/save_model_weights.md)  
[`with_custom_object_scope()`](https://keras3.posit.co/reference/with_custom_object_scope.md)  
