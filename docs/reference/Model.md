# Subclass the base Keras `Model` Class

This is for advanced use cases where you need to subclass the base
`Model` type, e.g., you want to override the `train_step()` method.

If you just want to create or define a keras model, prefer
[`keras_model()`](https://keras3.posit.co/reference/keras_model.md) or
[`keras_model_sequential()`](https://keras3.posit.co/reference/keras_model_sequential.md).

If you just want to encapsulate some custom logic and state, and don't
need to customize training behavior (besides calling `self$add_loss()`
in the [`call()`](https://rdrr.io/r/base/call.html) method), prefer
[`Layer()`](https://keras3.posit.co/reference/Layer.md).

## Usage

``` r
Model(
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

## Symbols in scope

All R function custom methods (public and private) will have the
following symbols in scope:

- `self`: The custom class instance.

- `super`: The custom class superclass.

- `private`: An R environment specific to the class instance. Any
  objects assigned here are invisible to the Keras framework.

- `__class__` and `as.symbol(classname)`: the custom class type object.

## See also

[`active_property()`](https://keras3.posit.co/reference/active_property.md)
(e.g., for a `metrics` property implemented as a function).
