# Preprocessing and postprocessing utilities

These functions are used to preprocess and postprocess inputs and
outputs of Keras applications.

## Usage

``` r
application_preprocess_inputs(model, x, ..., data_format = NULL)

application_decode_predictions(model, preds, top = 5L, ...)
```

## Arguments

- model:

  A Keras model initialized using any `application_` function.

- x:

  A batch of inputs to the model. If `x` is missing, then the
  `preprocess_input` function appropriate for `model` is returned.

- ...:

  Additional arguments passed to the preprocessing or decoding function.

- data_format:

  Optional data format of the image tensor/array. `NULL` means the
  global setting
  [`config_image_data_format()`](https://keras3.posit.co/dev/reference/config_image_data_format.md)
  is used (unless you changed it, it uses `"channels_last"`). Defaults
  to `NULL`.

- preds:

  A batch of outputs from the model.

- top:

  The number of top predictions to return.

## Value

- A list of decoded predictions in case of
  `application_decode_predictions()`.

- A batch of preprocessed inputs in case of
  `application_preprocess_inputs()`.

## Functions

- `application_preprocess_inputs()`: Pre-process inputs to be used in
  the model

- `application_decode_predictions()`: Decode predictions from the model

## Examples

``` r
if (FALSE) { # \dontrun{
model <- application_convnext_tiny()

inputs <- random_normal(c(32, 224, 224, 3))
processed_inputs <- application_preprocess_inputs(model, inputs)

preds <- random_normal(c(32, 1000))
decoded_preds <- application_decode_predictions(model, preds)

} # }
```
