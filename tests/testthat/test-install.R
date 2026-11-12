test_that("use_backend allows NumPy 2 on Windows", {
  requirements <- callr::r(function() {
    library(keras3)
    testthat::local_mocked_bindings(
      get_os = function() "Windows",
      .package = "keras3"
    )
    reticulate::py_require("numpy>=2", python_version = "3.13")
    keras3::use_backend("tensorflow", gpu = FALSE)
    reticulate::py_require()
  })

  expect_true("tensorflow" %in% requirements$packages)
  expect_true("numpy>=2" %in% requirements$packages)
  expect_false("numpy<2" %in% requirements$packages)
})

test_that("install_keras leaves NumPy version selection to its dependencies", {
  installed_packages <- character()
  testthat::local_mocked_bindings(
    virtualenv_create = function(...) NULL,
    py_install = function(packages, ...) {
      installed_packages <<- c(installed_packages, packages)
    },
    .package = "reticulate"
  )

  install_keras(backend = "tensorflow", gpu = FALSE, restart_session = FALSE)
  expect_true("keras==3.*" %in% installed_packages)
  expect_false("numpy<2" %in% installed_packages)
})

test_that("install_keras requests TensorFlow CUDA dependencies on Linux", {
  installed_packages <- character()
  testthat::local_mocked_bindings(
    is_linux = function() TRUE,
    is_mac_arm64 = function() FALSE,
    configure_nvidia_symlinks = function(...) NULL,
    configure_ptxas_symlink = function(...) NULL,
    .package = "keras3"
  )
  testthat::local_mocked_bindings(
    virtualenv_create = function(...) NULL,
    py_install = function(packages, ...) {
      installed_packages <<- c(installed_packages, packages)
    },
    .package = "reticulate"
  )

  install_keras(backend = "tensorflow", gpu = TRUE, restart_session = FALSE)
  expect_true("tensorflow[and-cuda]" %in% installed_packages)
})
