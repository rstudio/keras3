# Export the model as an artifact for inference.

This method lets you export a model to a lightweight artifact that
contains the model's forward pass only (its
[`call()`](https://rdrr.io/r/base/call.html) method). For TensorFlow
SavedModel artifacts, the forward pass is registered under the name
`serve()` and can be served via, for example, TF-Serving.

The original code of the model (including any custom layers you may have
used) is *no longer* necessary to reload the artifact – it is entirely
standalone.

**Note:** This feature is currently supported only with TensorFlow, JAX
and Torch backends.

**Note:** Be aware that the exported artifact may contain information
from the local file system when using `format = "onnx"`,
`verbose = TRUE` and Torch backend.

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

  Additional backend- or format-specific export options:

  - `is_static`: Optional boolean specific to the JAX backend and
    `format = "tf_saved_model"`. Indicates whether `fn` is static. Set
    to `FALSE` if `fn` involves state updates, such as RNG seeds and
    counters.

  - `jax2tf_kwargs`: Optional dictionary specific to the JAX backend and
    `format = "tf_saved_model"`. Arguments for
    [`jax2tf.convert`](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md).
    If `native_serialization` and `polymorphic_shapes` are not provided,
    they are computed automatically.

  - `opset_version`: Optional integer specific to `format = "onnx"` that
    specifies the ONNX opset version.

  - LiteRT-specific options. With the TensorFlow backend these are
    passed to the TensorFlow Lite converter and include `optimizations`,
    `representative_dataset`, `experimental_new_quantizer`,
    `allow_custom_ops`, and `enable_select_tf_ops`. With the PyTorch
    backend, options include `optimizations` and installed
    `litert_torch.convert()` keyword arguments such as `strict_export`,
    `dynamic_shapes`, `lightweight_conversion`, `enable_x64`,
    `runtime_constant_folding`, and `quant_config`.

  - PyTorch export options specific to `format = "torch"`, passed to
    `torch.export.export`, including `strict`, `dynamic_shapes`,
    `prefer_deferred_runtime_asserts_over_guards`, and
    `preserve_module_call_signature`.

- format:

  string. The export format. Supported values: `"tf_saved_model"`,
  `"onnx"`, `"openvino"`, `"litert"`, and `"torch"`. Defaults to
  `"tf_saved_model"`.

- verbose:

  Bool. Whether to print a message during export. Defaults to `NULL`,
  which uses the default value set by different backends and formats.

- input_signature:

  Optional. Specifies the shape and dtype of the model inputs. Can be a
  structure of `keras.InputSpec`, `tf.TensorSpec`,
  `backend.KerasTensor`, or backend tensor. If not provided, it will be
  automatically computed. Defaults to `NULL`. With `format = "litert"`
  and the PyTorch backend, dynamic input shapes are not supported. Any
  dynamic dimensions are automatically replaced with `1`, which may
  cause runtime failures for other shapes. Explicitly pass a fixed
  static `input_signature` matching the maximum runtime shape and pad
  inputs to that shape at runtime.

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

Here's how to export a LiteRT (TFLite) artifact for inference.

    model |> export_savedmodel("path/to/model.tflite", format = "litert")

    tf <- reticulate::import("tensorflow")
    interpreter <- tf$lite$Interpreter(model_path = "path/to/model.tflite")
    interpreter$allocate_tensors()
    interpreter$set_tensor(interpreter$get_input_details()[[1]]$index, input_data)
    interpreter$invoke()
    output_data <- interpreter$get_tensor(
      interpreter$get_output_details()[[1]]$index
    )

Here's how to export a PyTorch `ExportedProgram` for inference.

    # Export the model as a PyTorch ExportedProgram
    model |> export_savedmodel("path/to/model.pt2", format = "torch")

    # Load the artifact in a different process/environment
    torch <- reticulate::import("torch")
    loaded_program <- torch$export$load("path/to/model.pt2")
    predictions <- loaded_program$module()(input_tensor)

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
