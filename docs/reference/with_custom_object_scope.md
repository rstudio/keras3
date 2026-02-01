# Provide a scope with mappings of names to custom objects

Provide a scope with mappings of names to custom objects

## Usage

``` r
with_custom_object_scope(objects, expr)
```

## Arguments

- objects:

  Named list of objects

- expr:

  Expression to evaluate

## Value

The result from evaluating `expr` within the custom object scope.

## Details

There are many elements of Keras models that can be customized with user
objects (e.g. losses, metrics, regularizers, etc.). When loading saved
models that use these functions you typically need to explicitly map
names to user objects via the `custom_objects` parameter.

The `with_custom_object_scope()` function provides an alternative that
lets you create a named alias for a user object that applies to an
entire block of code, and is automatically recognized when loading saved
models.

## Examples

    # define custom metric
    metric_top_3_categorical_accuracy <-
      custom_metric("top_3_categorical_accuracy", function(y_true, y_pred) {
        metric_top_k_categorical_accuracy(y_true, y_pred, k = 3)
      })

    with_custom_object_scope(c(top_k_acc = sparse_top_k_cat_acc), {

      # ...define model...

      # compile model (refer to "top_k_acc" by name)
      model |> compile(
        loss = "binary_crossentropy",
        optimizer = optimizer_nadam(),
        metrics = c("top_k_acc")
      )

      # save the model
      model |> save_model("my_model.keras")

      # loading the model within the custom object scope doesn't
      # require explicitly providing the custom_object
      reloaded_model <- load_model("my_model.keras")
    })

## See also

Other saving and loading functions:  
[`export_savedmodel.keras.src.models.model.Model()`](https://keras3.posit.co/reference/export_savedmodel.keras.src.models.model.Model.md)  
[`layer_tfsm()`](https://keras3.posit.co/reference/layer_tfsm.md)  
[`load_model()`](https://keras3.posit.co/reference/load_model.md)  
[`load_model_weights()`](https://keras3.posit.co/reference/load_model_weights.md)  
[`register_keras_serializable()`](https://keras3.posit.co/reference/register_keras_serializable.md)  
[`save_model()`](https://keras3.posit.co/reference/save_model.md)  
[`save_model_config()`](https://keras3.posit.co/reference/save_model_config.md)  
[`save_model_weights()`](https://keras3.posit.co/reference/save_model_weights.md)  

Other serialization utilities:  
[`deserialize_keras_object()`](https://keras3.posit.co/reference/deserialize_keras_object.md)  
[`get_custom_objects()`](https://keras3.posit.co/reference/get_custom_objects.md)  
[`get_registered_name()`](https://keras3.posit.co/reference/get_registered_name.md)  
[`get_registered_object()`](https://keras3.posit.co/reference/get_registered_object.md)  
[`register_keras_serializable()`](https://keras3.posit.co/reference/register_keras_serializable.md)  
[`serialize_keras_object()`](https://keras3.posit.co/reference/serialize_keras_object.md)  
