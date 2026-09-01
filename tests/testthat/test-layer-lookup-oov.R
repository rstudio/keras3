test_that("lookup layers preserve OOV hashing configuration", {
  skip_if_no_keras("3.14.0")

  integer_config <- layer_integer_lookup(
    vocabulary = 1:3,
    num_oov_indices = 2L,
    oov_method = "farmhash",
    salt = c(11L, 13L)
  )$get_config()
  string_config <- layer_string_lookup(
    vocabulary = c("a", "b"),
    num_oov_indices = 2L,
    salt = c(11L, 13L)
  )$get_config()

  expect_identical(integer_config$oov_method, "farmhash")
  expect_identical(integer_config$salt, c(11L, 13L))
  expect_identical(string_config$salt, c(11L, 13L))
})
