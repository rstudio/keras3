# Install Keras

This function will install Keras along with a selected backend,
including all Python dependencies.

## Usage

``` r
install_keras(
  envname = "r-keras",
  ...,
  extra_packages = c("scipy", "pandas", "Pillow", "pydot", "ipython",
    "tensorflow_datasets"),
  python_version = ">=3.9,<=3.11",
  backend = c("tensorflow", "jax"),
  gpu = NA,
  restart_session = TRUE
)
```

## Arguments

- envname:

  Name of or path to a Python virtual environment

- ...:

  reserved for future compatibility.

- extra_packages:

  Additional Python packages to install alongside Keras

- python_version:

  Passed on to
  [`reticulate::virtualenv_starter()`](https://rstudio.github.io/reticulate/reference/virtualenv-tools.html)

- backend:

  Which backend(s) to install. Accepted values include `"tensorflow"`,
  `"jax"` and `"torch"`

- gpu:

  whether to install a GPU capable version of the backend.

- restart_session:

  Whether to restart the R session after installing (note this will only
  occur within RStudio).

## Value

No return value, called for side effects.

## See also

[`tensorflow::install_tensorflow()`](https://rdrr.io/pkg/tensorflow/man/install_tensorflow.html)
