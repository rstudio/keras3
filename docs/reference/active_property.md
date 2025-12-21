# Create an active property class method

Create an active property class method

## Usage

``` r
active_property(fn)
```

## Arguments

- fn:

  An R function

## Value

`fn`, with an additional R attribute that will cause `fn` to be
converted to an active property when being converted to a method of a
custom subclass.

## Example

    layer_foo <- Model("Foo", ...,
      metrics = active_property(function() {
        list(self$d_loss_metric,
             self$g_loss_metric)
      }))
