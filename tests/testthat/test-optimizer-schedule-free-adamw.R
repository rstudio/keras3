test_that("schedule-free AdamW exposes the Keras optimizer", {
  skip_if_no_keras("3.15.1")

  optimizer <- optimizer_schedule_free_adam_w(
    learning_rate = 0.01,
    warmup_steps = 2
  )

  expect_s3_class(
    optimizer,
    "keras.src.optimizers.schedule_free_adamw.ScheduleFreeAdamW"
  )
  expect_equal(optimizer$warmup_steps, 2L)
})
