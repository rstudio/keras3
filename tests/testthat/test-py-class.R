

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


test_that("%py_class% can be lazy about initing python", {
  res <- callr::r(function() {
    library(keras)
    # pretend we're in a package
    options("topLevelEnvironment" = environment())

    MyClass %py_class% {
      initialize <- function(x) {
        print("Hi from MyClass$initialize()!")
        self$x <- x
      }
      my_method <- function() {
        self$x
      }
    }

    if (reticulate:::is_python_initialized())
      stop()

    MyClass2(MyClass) %py_class% {
      "This will be a __doc__ string for MyClass2"

      initialize <- function(...) {
        "This will be the __doc__ string for the MyClass2.__init__() method"
        print("Hi from MyClass2$initialize()!")
        super$initialize(...)
      }
    }

    if (reticulate:::is_python_initialized())
      stop()

    my_class_instance2 <- MyClass2(42)
    my_class_instance2$my_method()
  })

  expect_equal(res, 42)
})


test_that("%py_class% initialize", {
  # check that single expression init functions defined with `{` work

  NaiveSequential %py_class% {
    initialize <- function(layers)
      self$layers <- layers
  }

  x <- NaiveSequential(list(1, "2", 3))
  expect_identical(x$layers, list(1, "2", 3))

})
