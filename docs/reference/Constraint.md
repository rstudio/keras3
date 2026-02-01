# Define a custom `Constraint` class

Base class for weight constraints.

A `Constraint()` instance works like a stateless function. Users who
subclass the `Constraint` class should override the
[`call()`](https://rdrr.io/r/base/call.html) method, which takes a
single weight parameter and return a projected version of that parameter
(e.g. normalized or clipped). Constraints can be used with various Keras
layers via the `kernel_constraint` or `bias_constraint` arguments.

Here's a simple example of a non-negative weight constraint:

    constraint_nonnegative <- Constraint("NonNegative",
      call = function(w) {
        w * op_cast(w >= 0, dtype = w$dtype)
      }
    )
    weight <- op_convert_to_tensor(c(-1, 1))
    constraint_nonnegative()(weight)

    ## tf.Tensor([-0.  1.], shape=(2), dtype=float64)

Usage in a layer:

    layer_dense(units = 4, kernel_constraint = constraint_nonnegative())

    ## <Dense name=dense, built=False>
    ##  signature: (*args, **kwargs)

## Usage

``` r
Constraint(
  classname,
  call = NULL,
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

- call:

      \(w)

  Applies the constraint to the input weight variable.

  By default, the inputs weight variable is not modified. Users should
  override this method to implement their own projection function.

  Args:

  - `w`: Input weight variable.

  Returns: Projected variable (by default, returns unmodified inputs).

- get_config:

      \()

  Function that returns a named list of the object config.

  A constraint config is a named list (JSON-serializable) that can be
  used to reinstantiate the same object (via
  `do.call(<constraint_class>, <config>)`).

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

A function that returns `Constraint` instances, similar to the builtin
constraint functions like
[`constraint_maxnorm()`](https://keras3.posit.co/reference/constraint_maxnorm.md).

## Symbols in scope

All R function custom methods (public and private) will have the
following symbols in scope:

- `self`: The custom class instance.

- `super`: The custom class superclass.

- `private`: An R environment specific to the class instance. Any
  objects assigned here are invisible to the Keras framework.

- `__class__` and `as.symbol(classname)`: the custom class type object.

## See also

Other constraints:  
[`constraint_maxnorm()`](https://keras3.posit.co/reference/constraint_maxnorm.md)  
[`constraint_minmaxnorm()`](https://keras3.posit.co/reference/constraint_minmaxnorm.md)  
[`constraint_nonneg()`](https://keras3.posit.co/reference/constraint_nonneg.md)  
[`constraint_unitnorm()`](https://keras3.posit.co/reference/constraint_unitnorm.md)  
