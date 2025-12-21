# LearningRateSchedule

`new_learning_rate_schedule_class()` is an alias for
[`LearningRateSchedule()`](https://keras3.posit.co/reference/LearningRateSchedule.md).
See
`?`[`LearningRateSchedule()`](https://keras3.posit.co/reference/LearningRateSchedule.md)
for the full documentation.

## Usage

``` r
new_learning_rate_schedule_class(
  classname,
  call = NULL,
  initialize = NULL,
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

- call, initialize, get_config:

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

A function that returns `LearningRateSchedule` instances, similar to the
built-in `learning_rate_schedule_*` family of functions.
