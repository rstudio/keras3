test_that("NumPy arrays dispatch through all.equal", {
  skip_if_not(reticulate::py_module_available("numpy"))

  np <- reticulate::import("numpy", convert = FALSE)
  x <- np$array(c(1, 2))

  expect_true(isTRUE(all.equal(x, np$array(c(1, 2)))))
  expect_false(isTRUE(all.equal(x, np$array(c(9, 8)))))
})
