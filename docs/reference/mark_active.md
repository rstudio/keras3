# active_property

`mark_active()` is an alias for
[`active_property()`](https://keras3.posit.co/reference/active_property.md).
See
`?`[`active_property()`](https://keras3.posit.co/reference/active_property.md)
for the full documentation.

## Usage

``` r
mark_active(fn)
```

## Arguments

- fn:

  An R function

## Value

`fn`, with an additional R attribute that will cause `fn` to be
converted to an active property when being converted to a method of a
custom subclass.
