
#' L1 and L2 regularization
#'
#' @param l Regularization factor.
#' @param l1 L1 regularization factor.
#' @param l2 L2 regularization factor.
#'
#' @export
regularizer_l1 <- function(l = 0.01) {
  keras$regularizers$l1(l = l)
}

#' @rdname regularizer_l1
#' @export
regularizer_l2 <- function(l = 0.01) {
  keras$regularizers$l2(l = l)
}

#' @rdname regularizer_l1
#' @export
regularizer_l1_l2 <- function(l1 = 0.01, l2 = 0.01) {
  keras$regularizers$l1_l2(l1 = l1, l2 = l2)
}
