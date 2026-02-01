# Layer

`new_layer_class()` is an alias for
[`Layer()`](https://keras3.posit.co/reference/Layer.md). See
`?`[`Layer()`](https://keras3.posit.co/reference/Layer.md) for the full
documentation.

## Usage

``` r
new_layer_class(
  classname,
  initialize = NULL,
  call = NULL,
  build = NULL,
  get_config = NULL,
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

- initialize, call, build, get_config:

  Recommended methods to implement. See description and details
  sections.

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

A composing layer constructor, with similar behavior to other layer
functions like
[`layer_dense()`](https://keras3.posit.co/reference/layer_dense.md). The
first argument of the returned function will be `object`, enabling
`initialize()`ing and [`call()`](https://rdrr.io/r/base/call.html) the
layer in one step while composing the layer with the pipe, like

    layer_foo <- Layer("Foo", ....)
    output <- inputs |> layer_foo()

To only `initialize()` a layer instance and not
[`call()`](https://rdrr.io/r/base/call.html) it, pass a missing or
`NULL` value to `object`, or pass all arguments to `initialize()` by
name.

    layer <- layer_dense(units = 2, activation = "relu")
    layer <- layer_dense(NULL, 2, activation = "relu")
    layer <- layer_dense(, 2, activation = "relu")

    # then you can call() the layer in a separate step
    outputs <- inputs |> layer()
