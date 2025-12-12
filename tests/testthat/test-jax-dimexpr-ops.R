test_that("DimExpr Ops keeps symbolic dims when R uses double scalars", {
  skip_if_not(reticulate::py_module_available("jax"))

  export <- reticulate::import("jax.export", convert = FALSE)
  dim <- export$symbolic_shape("n")[[1]]

  expr <- dim - 1 # 1 is a double in R; Ops method should coerce to int

  expect_s3_class(expr, "jax._src.export.shape_poly._DimExpr")
  expect_match(reticulate::py_str(expr), "n - 1")
  expect_false(any(grepl("Array", class(expr))))
})
