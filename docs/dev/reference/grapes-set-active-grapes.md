# Make an Active Binding

Make an Active Binding

## Usage

``` r
sym %<-active% value
```

## Arguments

- sym:

  symbol to bind

- value:

  A function to call when the value of `sym` is accessed.

## Value

`value`, invisibly

## Details

Active bindings defined in a
[`%py_class%`](https://keras3.posit.co/dev/reference/grapes-py_class-grapes.md)
are converted to `@property` decorated methods.

## See also

[`makeActiveBinding()`](https://rdrr.io/r/base/bindenv.html),
[`%py_class%`](https://keras3.posit.co/dev/reference/grapes-py_class-grapes.md)

## Examples

``` r
set.seed(1234)
x %<-active% function(value) {
  message("Evaluating function of active binding")
  if(missing(value))
    runif(1)
  else
   message("Received: ", value)
}
x
#> Evaluating function of active binding
#> [1] 0.1137034
x
#> Evaluating function of active binding
#> [1] 0.6222994
x <- "foo"
#> Evaluating function of active binding
#> Received: foo
x <- "foo"
#> Evaluating function of active binding
#> Received: foo
x
#> Evaluating function of active binding
#> [1] 0.6092747
rm(x) # cleanup
```
