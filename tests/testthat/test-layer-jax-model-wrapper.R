test_that("JAX model wrappers preserve native serialization platforms", {
  skip_if_no_keras("3.14.0")
  skip_if_not(reticulate::py_module_available("jax"))

  call_fn <- reticulate::py_eval(
    "lambda params, inputs: inputs",
    convert = FALSE
  )
  layer <- layer_jax_model_wrapper(
    call_fn = call_fn,
    params = list(),
    native_serialization_platforms = c("cpu", "tpu")
  )

  expect_identical(
    layer$jax2tf_native_serialization_platforms,
    c("cpu", "tpu")
  )
})
