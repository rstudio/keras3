test_that("distributed_get_device_count reports CPU devices", {
  skip_if_no_keras("3.15.1")

  expect_named(formals(distributed_get_device_count), "device_type")

  if (!is_backend("jax"))
    skip("get_device_count is implemented by the JAX backend")

  count <- distributed_get_device_count("cpu")

  expect_type(count, "integer")
  expect_gte(count, 1L)
})
