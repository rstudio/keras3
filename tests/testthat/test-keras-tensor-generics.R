context("KerasTensor generic methods")



# binary_arith_generics <- c("+", "-", "*", "/", "^", "%%", "%/%")
# binary_compr_generics <- c("==", "!=", "<", "<=", ">", ">=")
# binary_logic_generics <- c("&", "|")
#
# binary_generics <-
#   c(binary_arith_generics,
#     binary_compr_generics,
#     binary_logic_generics)
#
#
# unary_math_generics <- c(
#   "abs",
#   "sign",
#   "sqrt",
#   "floor",
#   "ceiling",
#   "round",
#
#   "log",
#   "log1p",
#   "log2",
#   "log10",
#
#   "exp",
#   "expm1",
#
#   "cos",
#   "sin",
#   "tan",
#
#   "sinpi",
#   "cospi",
#   "tanpi",
#
#   "acos",
#   "asin",
#   "atan",
#
#   "lgamma",
#   "digamma"
# )
# unary_shape_generics <- c("dim", "length")
#
# unary_complex_generics <- c("Re", "Im", "Conj", "Arg", "Mod")
#
# unary_fn_generics <- c(
#   unary_math_generics,
#   unary_complex_generics,
#   unary_complex_generics
# )
#
# unary_operator_generics <- c("!", "-", "+")

# glue::glue("expect_keras_tensor(kt {binary_generics} kt)") %>%  clipr::write_clip()
# glue::glue("expect_keras_tensor({unary_operator_generics} kt)") %>%  clipr::write_clip()
# glue::glue("expect_keras_tensor({unary_fn_generics}(kt))") %>%  clipr::write_clip()



test_that("tensor generics work with KerasTensor", {
  kt <- layer_input(list())
  kt_cls <- class(kt)[1]

  # if (!any(grepl("KerasTensor", kt_cls)))
  #   skip("Don't have KerasTensors")

  kt_lgl <- layer_input(list(), dtype = "bool")
  expect_keras_tensor <- function(expr)
    expect_s3_class(expr, kt_cls)

  expect_keras_tensor(kt + kt)
  expect_keras_tensor(kt - kt)
  expect_keras_tensor(kt * kt)
  expect_keras_tensor(kt / kt)
  expect_keras_tensor(kt ^ kt)
  expect_keras_tensor(kt %% kt)
  expect_keras_tensor(kt %/% kt)
  expect_keras_tensor(kt == kt)
  expect_keras_tensor(kt != kt)
  expect_keras_tensor(kt < kt)
  expect_keras_tensor(kt <= kt)
  expect_keras_tensor(kt > kt)
  expect_keras_tensor(kt >= kt)

  expect_keras_tensor(kt_lgl & kt_lgl)
  expect_keras_tensor(kt_lgl | kt_lgl)

  expect_keras_tensor(!kt_lgl)
  expect_keras_tensor(-kt)
  expect_keras_tensor(+kt)

  expect_keras_tensor(abs(kt))
  expect_keras_tensor(sign(kt))
  expect_keras_tensor(sqrt(kt))
  expect_keras_tensor(floor(kt))
  expect_keras_tensor(ceiling(kt))
  expect_keras_tensor(round(kt))

  expect_keras_tensor(log(kt))
  expect_keras_tensor(log(kt, base = pi))
  expect_keras_tensor(log(kt, base = tf$convert_to_tensor(pi)))
  expect_keras_tensor(log(kt, base = kt))

  expect_keras_tensor(log1p(kt))
  expect_keras_tensor(log2(kt))
  expect_keras_tensor(log10(kt))
  expect_keras_tensor(exp(kt))
  expect_keras_tensor(expm1(kt))
  expect_keras_tensor(cos(kt))
  expect_keras_tensor(sin(kt))
  expect_keras_tensor(tan(kt))
  expect_keras_tensor(sinpi(kt))
  expect_keras_tensor(cospi(kt))
  expect_keras_tensor(tanpi(kt))
  expect_keras_tensor(acos(kt))
  expect_keras_tensor(asin(kt))
  expect_keras_tensor(atan(kt))
  expect_keras_tensor(lgamma(kt))
  expect_keras_tensor(digamma(kt))
  expect_keras_tensor(Re(kt))
  expect_keras_tensor(Im(kt))
  expect_keras_tensor(Conj(kt))
  expect_keras_tensor(Arg(kt))
  expect_keras_tensor(Mod(kt))
  expect_keras_tensor(Re(kt))
  expect_keras_tensor(Im(kt))
  expect_keras_tensor(Conj(kt))
  expect_keras_tensor(Arg(kt))
  expect_keras_tensor(Mod(kt))

  expect_keras_tensor(kt[1])


})
