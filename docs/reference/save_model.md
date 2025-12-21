# Saves a model as a `.keras` file.

Saves a model as a `.keras` file.

## Usage

``` r
save_model(model, filepath = NULL, overwrite = FALSE, zipped = NULL, ...)
```

## Arguments

- model:

  A keras model.

- filepath:

  string, Path where to save the model. Must end in `.keras`.

- overwrite:

  Whether we should overwrite any existing model at the target location,
  or instead ask the user via an interactive prompt.

- zipped:

  Whether to save the model as a zipped `.keras` archive (default when
  saving locally), or as an unzipped directory (default when saving on
  the Hugging Face Hub).

- ...:

  For forward/backward compatability.

## Value

If `filepath` is provided, then this function is called primarily for
side effects, and `model` is returned invisibly. If `filepath` is not
provided or `NULL`, then the serialized model is returned as an R raw
vector.

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

The saved `.keras` file is a `zip` archive that contains:

- The model's configuration (architecture)

- The model's weights

- The model's optimizer's state (if any)

Thus models can be reinstantiated in the exact same state.

    zip::zip_list("model.keras")[, "filename"]

    ## [1] "metadata.json"    "config.json"      "model.weights.h5"

## See also

[`load_model()`](https://keras3.posit.co/reference/load_model.md)

Other saving and loading functions:  
[`export_savedmodel.keras.src.models.model.Model()`](https://keras3.posit.co/reference/export_savedmodel.keras.src.models.model.Model.md)  
[`layer_tfsm()`](https://keras3.posit.co/reference/layer_tfsm.md)  
[`load_model()`](https://keras3.posit.co/reference/load_model.md)  
[`load_model_weights()`](https://keras3.posit.co/reference/load_model_weights.md)  
[`register_keras_serializable()`](https://keras3.posit.co/reference/register_keras_serializable.md)  
[`save_model_config()`](https://keras3.posit.co/reference/save_model_config.md)  
[`save_model_weights()`](https://keras3.posit.co/reference/save_model_weights.md)  
[`with_custom_object_scope()`](https://keras3.posit.co/reference/with_custom_object_scope.md)  
