# Model

`new_model_class()` is an alias for
[`Model()`](https://keras3.posit.co/reference/Model.md). See
`?`[`Model()`](https://keras3.posit.co/reference/Model.md) for the full
documentation.

## Usage

``` r
new_model_class(
  classname,
  initialize = NULL,
  call = NULL,
  train_step = NULL,
  predict_step = NULL,
  test_step = NULL,
  compute_loss = NULL,
  compute_metrics = NULL,
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

- initialize, call, train_step, predict_step, test_step, compute_loss,
  compute_metrics:

  Optional methods that can be overridden.

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

A model constructor function, which you can call to create an instance
of the new model type.
