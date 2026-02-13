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
  detect GPU availability on Linux. On macOS and Windows it defaults to
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

Note that macOS packages like `tensorflow-metal` and `jax-metal` that
purportedly enabled GPU usage on M-series macs all are currently broken
and seemingly abandoned.

There is experimental support for changing the backend after keras has
initialized with
[`config_set_backend()`](https://keras3.posit.co/dev/reference/config_set_backend.md).
Usage of `config_set_backend` is generally not recommended for regular
workflow—restarting the R session is the only reliable way to change the
backend.

    library(keras3)
    use_backend("tensorflow")
