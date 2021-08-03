# closet with future and past utils not currently in use

#' @importFrom rlang names2
named_list <- function(...) {
    exprs <- eval(substitute(alist(...)))
    vals <- list(...)
    nms <- names2(vals)

    missing <- nms == ""
    if (all(!missing))
      return(vals)

    deparse2 <- function(x) paste(deparse(x, 500L), collapse = "")
    defaults <- vapply(exprs[missing], deparse2, "", USE.NAMES = FALSE)
    names(vals)[missing] <- defaults
    vals
}

drop_nulls <- function(x, i = NULL) {
  if(is.null(i))
    x[!vapply(x, is.null, FALSE, USE.NAMES = FALSE)]
  else {
    drop <- rep(FALSE, length(x))
    drop[i] <- vapply(x[i], is.null, FALSE, USE.NAMES = FALSE)
    x[drop]
  }
}

# conflict with utils::zip, maybe another name?
zip <- function(..., simplify = TRUE)
  .mapply(if(simplify) c else list, list(...), NULL)
