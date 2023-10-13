

# `_COMMON_` <- NULL

library_stable <-
  function(package,
           lifecycle_exclude = c(
             "superseded",
             "deprecated",
             "questioning",
             "defunct",
             "experimental",
             "soft-deprecated",
             "retired"
           ), ...,
           exclude = NULL) {

  package <- deparse1(substitute(package))
  exclude <- lifecycle::pkg_lifecycle_statuses(
    package, lifecycle_exclude)$fun |> c(exclude)

  # message(package, ", excludeing: ", paste0(exclude, collapse = ", "))
  library(package, character.only = TRUE, ...,
          exclude = exclude)
}

library_stable(readr)
library_stable(stringr)
library_stable(tibble)
library_stable(tidyr)
library_stable(dplyr, warn.conflicts = FALSE)
library_stable(rlang) #, exclude = c("splice"))
library_stable(purrr) #, exclude)
library_stable(stringr)
library_stable(glue)

library(envir)
library(commafree)
library(magrittr, include.only = c("%>%", "%<>%"))
library(reticulate)
library(assertthat, include.only = c("assert_that"))

# ---- utils ----
import_from(TKutils, `%error%`)

is_scalar <- function(x) identical(length(x), 1L)

replace_val <- function (x, old, new) {
  if (!is_scalar(new))
    stop("Unexpected length of replacement value in replace_val().\n",
         "`new` must be length 1, not ", length(new))
  x[x %in% old] <- new
  x
}

py_is <- function(x, y) identical(py_id(x), py_id(y))

str_split1_on_first <- function(x, pattern, ...) {
  stopifnot(length(x) == 1, is.character(x))
  regmatches(x, regexpr(pattern, x, ...), invert = TRUE)[[1L]]
}

str_flatten_lines <- function(...) {
  stringr::str_flatten(c(...), collapse = "\n")
}

str_split_lines <- function(...) {
  x <- c(...)
  if(length(x) > 1)
    x <- str_flatten_lines(x)
  stringr::str_split_1(x, "\n")
}


`append<-` <- function(x, after = length(x), value) {
  append(x, value, after)
}

unwhich <- function(x, len) {
  lgl <- logical(len)
  lgl[x] <- TRUE
  lgl
}

# ---- venv ----
# reticulate::virtualenv_remove("r-keras")
if(!virtualenv_exists("r-keras")) {
   install_keras()
}


use_virtualenv("r-keras")

inspect <- import("inspect")

# keras <- import("keras_core")
# keras <- import("tensorflow.keras")
keras <- import("keras")
local({
  `__main__` <- reticulate::import_main()
  `__main__`$keras <- keras
})

source_python("tools/common.py") # keras_class_type()
rm(r) # TODO: fix in reticulate, don't export r
