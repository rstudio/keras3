

test_that("R6 pyclasses respect convert=FALSE", {
  # test that a python dictionary is modified in place if passed as an argument to
  # a class method.

  d <- py_eval("{}", FALSE)

  r_cls <- R6Class(
    "r_class",
    public = list(
      a_method = function(x = NULL) {
        expect_same_pyobj(x, d)
        x$update(list("foo" = NULL))
      }
    ))

  py_cls <- r_to_py(r_cls, convert = FALSE)
  py_inst <- py_cls()
  py_inst$a_method(d)

  expect_identical(py_to_r(d), list("foo" = NULL))
})


test_that("R6 pyclasses respect convert=TRUE", {

  d <- py_eval("{}", FALSE)

  r_cls <- R6Class(
    "r_class",
    public = list(
      a_method = function(x = NULL) {
        expect_type(x, "list")
        x[["foo"]] <- "bar"
        x
      }
    ))

  py_cls <- r_to_py(r_cls, convert = TRUE)
  py_inst <- py_cls()
  ret <- py_inst$a_method(d)

  expect_identical(ret, list(foo = "bar"))
  expect_identical(py_to_r(d), py_eval("{}"))
})
