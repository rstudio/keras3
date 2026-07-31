test_that("quantization config constructors expose public APIs", {
  skip_if_no_keras("3.15.1")

  expect_named(
    formals(quantizer_quantization_config),
    c("weight_quantizer", "activation_quantizer")
  )
  expect_s3_class(
    quantizer_float8_quantization_config(),
    "keras.src.quantizers.quantization_config.Float8QuantizationConfig"
  )
  expect_s3_class(
    quantizer_int4_quantization_config(),
    "keras.src.quantizers.quantization_config.Int4QuantizationConfig"
  )
  expect_s3_class(
    quantizer_int8_quantization_config(),
    "keras.src.quantizers.quantization_config.Int8QuantizationConfig"
  )
})

test_that("calibration quantization config constructors accept R values", {
  skip_if_no_keras("3.15.1")

  tokenizer <- function(x) x

  expect_s3_class(
    quantizer_awq_config(list("calibration text"), tokenizer),
    "keras.src.quantizers.awq_config.AWQConfig"
  )
  expect_s3_class(
    quantizer_gptq_config(list("calibration text"), tokenizer),
    "keras.src.quantizers.gptq_config.GPTQConfig"
  )
})

test_that("grouped asymmetric quantization returns values by group", {
  skip_if_no_keras("3.15.1")

  inputs <- op_reshape(op_arange(16, dtype = "float32"), c(4, 4))

  result <- quantizer_abs_max_quantize_grouped_with_zero_point(
    inputs,
    block_size = 2
  )

  expect_length(result, 3)
  expect_equal(unlist(shape(result[[1]])), c(4L, 4L))
  expect_equal(unlist(shape(result[[2]])), c(2L, 4L))
  expect_equal(unlist(shape(result[[3]])), c(2L, 4L))
})
