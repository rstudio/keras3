#' @md
#' 
#' @section Custom constraints:
#' 
#' You can implement your own constraint functions in R. A custom 
#' constraint is an R function that takes weights (`w`) as input
#' and returns modified weights. Note that keras [backend()] tensor
#' functions (e.g. `k_greater_equal()`) should be used in the 
#' implementation of custom constraints. For example:
#' 
#' ```
#' nonneg_constraint <- function(w) {
#'   w * k_cast(k_greater_equal(w, 0), k_floatx())
#' }
#' ```
#' 
