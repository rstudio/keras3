# Configure a Keras backend

Configure a Keras backend

## Usage

``` r
use_backend(backend, gpu = NA)
```

## Arguments

- backend:

  string, can be `"tensorflow"`, `"jax"`, `"numpy"`, or `"torch"`.

- gpu:

  bool, whether to use the GPU. If `NA` (default), it will attempt to
  detect GPU availability on Linux. On M-series Macs, it defaults to
  `FALSE` for TensorFlow and `TRUE` for JAX. On Windows, it defaults to
  `FALSE`.

## Value

Called primarily for side effects. Returns the provided `backend`,
invisibly.

## Details

These functions allow configuring which backend keras will use. Note
that only one backend can be configured at a time.

The function should be called after
[`library(keras3)`](https://keras3.posit.co/) and before calling other
functions within the package (see below for an example).

There is experimental support for changing the backend after keras has
initialized. using
[`config_set_backend()`](https://keras3.posit.co/dev/reference/config_set_backend.md).

    library(keras3)
    use_backend("tensorflow")
