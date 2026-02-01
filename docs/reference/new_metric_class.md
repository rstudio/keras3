# Metric

`new_metric_class()` is an alias for
[`Metric()`](https://keras3.posit.co/reference/Metric.md). See
`?`[`Metric()`](https://keras3.posit.co/reference/Metric.md) for the
full documentation.

## Usage

``` r
new_metric_class(
  classname,
  initialize = NULL,
  update_state = NULL,
  result = NULL,
  ...,
  public = list(),
  private = list(),
  inherit = NULL,
  parent_env = parent.frame()
)
```

## Arguments

- classname:

  String, the name of the custom class. (Conventionally, CamelCase).

- initialize, update_state, result:

  Recommended methods to implement. See description section.

- ..., public:

  Additional methods or public members of the custom class.

- private:

  Named list of R objects (typically, functions) to include in instance
  private environments. `private` methods will have all the same symbols
  in scope as public methods (See section "Symbols in Scope"). Each
  instance will have it's own `private` environment. Any objects in
  `private` will be invisible from the Keras framework and the Python
  runtime.

- inherit:

  What the custom class will subclass. By default, the base keras class.

- parent_env:

  The R environment that all class methods will have as a grandparent.

## Value

A function that returns `Metric` instances, similar to the builtin
metric functions.
