# Export the model as an artifact for inference.

(e.g. via TF-Serving).

**Note:** This can currently only be used with the TensorFlow or JAX
backends.

This method lets you export a model to a lightweight SavedModel artifact
that contains the model's forward pass only (its
[`call()`](https://rdrr.io/r/base/call.html) method) and can be served
via e.g. TF-Serving. The forward pass is registered under the name
`serve()` (see example below).

The original code of the model (including any custom layers you may have
used) is *no longer* necessary to reload the artifact – it is entirely
standalone.

**Note:** This feature is currently supported only with TensorFlow, JAX
and Torch backends.

**Note:** Be aware that the exported artifact may contain information
from the local file system when using `format="onnx"`, `verbose=TRUE`
and Torch backend.

## Usage

``` r
# S3 method for class 'keras.src.models.model.Model'
export_savedmodel(
  object,
  export_dir_base,
  ...,
  format = "tf_saved_model",
  verbose = NULL,
  input_signature = NULL
)
```

## Arguments

- object:

  A keras model.

- export_dir_base:

  string, file path where to save the artifact.

- ...:

  Additional keyword arguments:

  - Specific to the JAX backend and `format="tf_saved_model"`:

    - `is_static`: Optional `bool`. Indicates whether `fn` is static.
      Set to `FALSE` if `fn` involves state updates (e.g., RNG seeds and
      counters).

    - `jax2tf_kwargs`: Optional `dict`. Arguments for `jax2tf.convert`.
      See the documentation for
      [`jax2tf.convert`](https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md).
      If `native_serialization` and `polymorphic_shapes` are not
      provided, they will be automatically computed.

- format:

  string. The export format. Supported values: `"tf_saved_model"` and
  `"onnx"`. Defaults to `"tf_saved_model"`.

- verbose:

  Bool. Whether to print all the variables of the exported model.
  Defaults to `NULL`, which uses the default value set by different
  backends and formats.

- input_signature:

  Optional. Specifies the shape and dtype of the model inputs. Can be a
  structure of `keras.InputSpec`, `tf.TensorSpec`,
  `backend.KerasTensor`, or backend tensor. If not provided, it will be
  automatically computed. Defaults to `NULL`.

## Value

This is called primarily for the side effect of exporting `object`. The
first argument, `object` is also returned, invisibly, to enable usage
with the pipe.

## Examples

    # Create the artifact
    model |> tensorflow::export_savedmodel("path/to/location")

    # Later, in a different process/environment...
    library(tensorflow)
    reloaded_artifact <- tf$saved_model$load("path/to/location")
    predictions <- reloaded_artifact$serve(input_data)

    # see tfdeploy::serve_savedmodel() for serving a model over a local web api.

Here's how to export an ONNX for inference.

    # Export the model as a ONNX artifact
    model |> export_savedmodel("path/to/location", format = "onnx")

    # Load the artifact in a different process/environment
    onnxruntime <- reticulate::import("onnxruntime")
    ort_session <- onnxruntime$InferenceSession("path/to/location")
    input_data <- list(....)
    names(input_data) <- sapply(ort_session$get_inputs(), `[[`, "name")
    predictions <- ort_session$run(NULL, input_data)

## See also

Other saving and loading functions:  
[`layer_tfsm()`](https://keras3.posit.co/dev/reference/layer_tfsm.md)  
[`load_model()`](https://keras3.posit.co/dev/reference/load_model.md)  
[`load_model_weights()`](https://keras3.posit.co/dev/reference/load_model_weights.md)  
[`register_keras_serializable()`](https://keras3.posit.co/dev/reference/register_keras_serializable.md)  
[`save_model()`](https://keras3.posit.co/dev/reference/save_model.md)  
[`save_model_config()`](https://keras3.posit.co/dev/reference/save_model_config.md)  
[`save_model_weights()`](https://keras3.posit.co/dev/reference/save_model_weights.md)  
[`with_custom_object_scope()`](https://keras3.posit.co/dev/reference/with_custom_object_scope.md)  
