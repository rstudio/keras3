test_that("quantized dtype policy constructors expose public APIs", {
  skip_if_no_keras("3.15.1")

  policies <- list(
    dtype_policy_awq("awq/4/128"),
    dtype_policy_gptq("gptq/4/128"),
    dtype_policy_int4("int4/128")
  )

  expect_s3_class(policies[[1]], "keras.src.dtype_policies.dtype_policy.AWQDTypePolicy")
  expect_s3_class(policies[[2]], "keras.src.dtype_policies.dtype_policy.GPTQDTypePolicy")
  expect_s3_class(policies[[3]], "keras.src.dtype_policies.dtype_policy.Int4DTypePolicy")
})
