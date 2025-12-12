# Plot training history

Plots metrics recorded during training.

## Usage

``` r
# S3 method for class 'keras_training_history'
plot(
  x,
  y,
  metrics = NULL,
  method = c("auto", "ggplot2", "base"),
  smooth = getOption("keras.plot.history.smooth", TRUE),
  theme_bw = getOption("keras.plot.history.theme_bw", FALSE),
  ...
)
```

## Arguments

- x:

  Training history object returned from
  [`fit.keras.src.models.model.Model()`](https://keras3.posit.co/dev/reference/fit.keras.src.models.model.Model.md).

- y:

  Unused.

- metrics:

  One or more metrics to plot (e.g. `c('loss', 'accuracy')`). Defaults
  to plotting all captured metrics.

- method:

  Method to use for plotting. The default "auto" will use ggplot2 if
  available, and otherwise will use base graphics.

- smooth:

  Whether a loess smooth should be added to the plot, only available for
  the `ggplot2` method. If the number of epochs is smaller than ten, it
  is forced to false.

- theme_bw:

  Use
  [`ggplot2::theme_bw()`](https://ggplot2.tidyverse.org/reference/ggtheme.html)
  to plot the history in black and white.

- ...:

  Additional parameters to pass to the
  [`plot()`](https://rdrr.io/r/graphics/plot.default.html) method.

## Value

if `method == "ggplot2"`, the ggplot object is returned. If
`method == "base"`, then this function will draw to the graphics device
and return `NULL`, invisibly.
