test_that("core layers preserve quantization configurations", {
  skip_if_no_keras("3.15.1")

  quantization_config <- keras$quantizers$Int8QuantizationConfig()
  dense <- layer_dense(
    units = 2L,
    quantization_config = quantization_config
  )
  embedding <- layer_embedding(
    input_dim = 5L,
    output_dim = 2L,
    quantization_config = quantization_config
  )
  einsum <- layer_einsum_dense(
    equation = "ab,bc->ac",
    output_shape = 2L,
    gptq_unpacked_column_size = 8L,
    quantization_config = quantization_config
  )

  expect_identical(
    dense$get_config()$quantization_config$class_name,
    "Int8QuantizationConfig"
  )
  expect_identical(
    embedding$get_config()$quantization_config$class_name,
    "Int8QuantizationConfig"
  )
  expect_identical(
    einsum$get_config()$quantization_config$class_name,
    "Int8QuantizationConfig"
  )
  expect_identical(einsum$gptq_unpacked_column_size, 8L)
})
