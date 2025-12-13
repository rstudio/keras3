# Constrains the weights to be non-negative.

Constrains the weights to be non-negative.

## Usage

``` r
constraint_nonneg()
```

## Value

A `Constraint` instance, a callable that can be passed to layer
constructors or used directly by calling it with tensors.

## See also

- <https://keras.io/api/layers/constraints#nonneg-class>

Other constraints:  
[`Constraint()`](https://keras3.posit.co/dev/reference/Constraint.md)  
[`constraint_maxnorm()`](https://keras3.posit.co/dev/reference/constraint_maxnorm.md)  
[`constraint_minmaxnorm()`](https://keras3.posit.co/dev/reference/constraint_minmaxnorm.md)  
[`constraint_unitnorm()`](https://keras3.posit.co/dev/reference/constraint_unitnorm.md)  
