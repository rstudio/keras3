# Print a summary of a Keras Model

Print a summary of a Keras Model

## Usage

``` r
# S3 method for class 'keras.src.models.model.Model'
summary(object, ...)

# S3 method for class 'keras.src.models.model.Model'
format(
  x,
  line_length = min(getOption("width"), 180),
  positions = NULL,
  expand_nested = FALSE,
  show_trainable = NA,
  ...,
  layer_range = NULL,
  compact = TRUE
)

# S3 method for class 'keras.src.models.model.Model'
print(x, ...)
```

## Arguments

- object, x:

  Keras model instance

- ...:

  for [`summary()`](https://rdrr.io/r/base/summary.html) and
  [`print()`](https://rdrr.io/r/base/print.html), passed on to
  [`format()`](https://rdrr.io/r/base/format.html). For
  [`format()`](https://rdrr.io/r/base/format.html), passed on to
  `model$summary()`.

- line_length:

  Total length of printed lines

- positions:

  Relative or absolute positions of log elements in each line. If not
  provided, defaults to `c(0.33, 0.55, 0.67, 1.0)`.

- expand_nested:

  Whether to expand the nested models. If not provided, defaults to
  `FALSE`.

- show_trainable:

  Whether to show if a layer is trainable. If not provided, defaults to
  `FALSE`.

- layer_range:

  a list, tuple, or vector of 2 strings, which is the starting layer
  name and ending layer name (both inclusive) indicating the range of
  layers to be printed in summary. It also accepts regex patterns
  instead of exact name. In such case, start predicate will be the first
  element it matches to `layer_range[[1]]` and the end predicate will be
  the last element it matches to `layer_range[[1]]`. By default `NULL`
  which considers all layers of model.

- compact:

  Whether to remove white-space only lines from the model summary.
  (Default `TRUE`)

## Value

[`format()`](https://rdrr.io/r/base/format.html) returns a length 1
character vector. [`print()`](https://rdrr.io/r/base/print.html) returns
the model object invisibly.
[`summary()`](https://rdrr.io/r/base/summary.html) returns the output of
[`format()`](https://rdrr.io/r/base/format.html) invisibly after
printing it.

## Enabling color output in Knitr (RMarkdown, Quarto)

In order to enable color output in a quarto or rmarkdown document with
an html output format (include revealjs presentations), then you will
need to do the following in a setup chunk:

     ```{r setup, include = FALSE}
     options(cli.num_colors = 256)
     fansi::set_knit_hooks(knitr::knit_hooks)
     options(width = 75) # adjust as needed for format
     ```

## See also

Other model functions:  
[`get_config()`](https://keras3.posit.co/reference/get_config.md)  
[`get_layer()`](https://keras3.posit.co/reference/get_layer.md)  
[`get_state_tree()`](https://keras3.posit.co/reference/get_state_tree.md)  
[`keras_model()`](https://keras3.posit.co/reference/keras_model.md)  
[`keras_model_sequential()`](https://keras3.posit.co/reference/keras_model_sequential.md)  
[`pop_layer()`](https://keras3.posit.co/reference/pop_layer.md)  
[`set_state_tree()`](https://keras3.posit.co/reference/set_state_tree.md)  
