test_that("TerminateOnNaN can raise on invalid loss", {
  skip_if_no_keras("3.14.0")

  callback <- callback_terminate_on_nan(raise_error = TRUE)

  expect_error(
    callback$on_batch_end(0L, list(loss = NaN)),
    "NaN or Inf"
  )
})
