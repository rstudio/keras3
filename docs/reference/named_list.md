# Create a named list from arguments

Constructs a list from the provided arguments where all elements are
named. This wraps
[`rlang::dots_list()`](https://rlang.r-lib.org/reference/list2.html) but
changes two defaults:

- `.named` is set to `TRUE`

- `.homonyms` is set to `"error"`

## Usage

``` r
named_list(...)
```

## Arguments

- ...:

  Arguments to collect in a list. These dots are
  [dynamic](https://rlang.r-lib.org/reference/dyn-dots.html).

## Value

A named list.

## Details

Other parameters retain their defaults from
[`rlang::dots_list()`](https://rlang.r-lib.org/reference/list2.html):  

- `.ignore_empty = "trailing"`

- `.preserve_empty = FALSE`

- `.check_assign = FALSE`

## See also

[`rlang::dots_list()`](https://rlang.r-lib.org/reference/list2.html)
