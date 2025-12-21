# Plot a Keras model

Plot a Keras model

## Usage

``` r
# S3 method for class 'keras.src.models.model.Model'
plot(
  x,
  show_shapes = FALSE,
  show_dtype = FALSE,
  show_layer_names = FALSE,
  ...,
  rankdir = "TB",
  expand_nested = FALSE,
  dpi = getOption("keras.plot.model.dpi", 200L),
  layer_range = NULL,
  show_layer_activations = FALSE,
  show_trainable = NA,
  to_file = NULL
)
```

## Arguments

- x:

  A Keras model instance

- show_shapes:

  whether to display shape information.

- show_dtype:

  whether to display layer dtypes.

- show_layer_names:

  whether to display layer names.

- ...:

  passed on to Python `keras.utils.model_to_dot()`. Used for forward and
  backward compatibility.

- rankdir:

  a string specifying the format of the plot: `'TB'` creates a vertical
  plot; `'LR'` creates a horizontal plot. (argument passed to PyDot)

- expand_nested:

  Whether to expand nested models into clusters.

- dpi:

  Dots per inch. Increase this value if the image text appears
  excessively pixelated.

- layer_range:

  `list` containing two character strings, which is the starting layer
  name and ending layer name (both inclusive) indicating the range of
  layers for which the plot will be generated. It also accepts regex
  patterns instead of exact name. In such case, start predicate will be
  the first element it matches to `layer_range[1]` and the end predicate
  will be the last element it matches to `layer_range[2]`. By default
  `NULL` which considers all layers of model. Note that you must pass
  range such that the resultant subgraph must be complete.

- show_layer_activations:

  Display layer activations (only for layers that have an `activation`
  property).

- show_trainable:

  whether to display if a layer is trainable.

- to_file:

  File name of the plot image. If `NULL` (the default), the model is
  drawn on the default graphics device. Otherwise, a file is saved.

## Value

Nothing, called for it side effects.

## Raises

ValueError: if `plot(model)` is called before the model is built, unless
an `input_shape = ` argument was supplied to
[`keras_model_sequential()`](https://keras3.posit.co/reference/keras_model_sequential.md).

## Requirements

This function requires pydot and graphviz.

`pydot` is by default installed by
[`install_keras()`](https://keras3.posit.co/reference/install_keras.md),
but if you installed Keras by other means, you can install `pydot`
directly with:

    reticulate::py_install("pydot", pip = TRUE)

You can install graphviz directly from here:
<https://graphviz.gitlab.io/download/>

On most Linux platforms, can install graphviz via the package manager.
For example, on Ubuntu/Debian you can install with

    sudo apt install graphviz

On macOS you can install graphviz using `brew`:

    brew install graphviz

In a conda environment, you can install graphviz with:

    reticulate::conda_install(packages = "graphviz")
    # Restart the R session after install.
