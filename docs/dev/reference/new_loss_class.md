# Loss

`new_loss_class()` is an alias for
[`Loss()`](https://keras3.posit.co/dev/reference/Loss.md). See
`?`[`Loss()`](https://keras3.posit.co/dev/reference/Loss.md) for the
full documentation.

## Usage

``` r
new_loss_class(
  classname,
  call = NULL,
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

- call:

      function(y_true, y_pred)

  Method to be implemented by subclasses: Function that contains the
  logic for loss calculation using `y_true`, `y_pred`.

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

A function that returns `Loss` instances, similar to the builtin loss
functions.
