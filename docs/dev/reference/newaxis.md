# New axis

This is an alias for `NULL`. It is meant to be used in `[` on tensors,
to expand dimensions of a tensor

## Usage

``` r
newaxis
```

## Format

An object of class `NULL` of length 0.

## Details

    x <- op_convert_to_tensor(1:10)

    op_shape(x)
    op_shape(x[])
    op_shape(x[newaxis])
    op_shape(x@py[newaxis])
    op_shape(x@r[newaxis])

    op_shape(x[newaxis, .., newaxis])
    op_shape(x@py[newaxis, .., newaxis])
    op_shape(x@r[newaxis, .., newaxis])
