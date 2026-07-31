test_that("callback_orbax_checkpoint exposes the Orbax callback", {
  expect_true(is.function(callback_orbax_checkpoint))
  expect_named(
    formals(callback_orbax_checkpoint),
    c(
      "directory", "monitor", "verbose", "save_best_only", "mode",
      "save_freq", "initial_value_threshold", "max_to_keep",
      "save_on_background", "save_weights_only"
    )
  )
})
