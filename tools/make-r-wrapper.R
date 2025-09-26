#!/usr/bin/env Rscript

if (basename(getwd()) != "keras3") {
  stop("must be called from the keras3 package directory")
}
Sys.setenv("TF_CPP_MIN_LOG_LEVEL" = "3", UV_NO_PROGRESS = 1)
# sink(nullfile())
suppressWarnings(suppressMessages(
  envir::attach_source("tools/utils.R")
))
# sink()

# Example usage:
# tools/make-r-wrapper.R keras.ops.add
for (py_expr in commandArgs(TRUE)) {
  writeLines(c("", mk_export(py_expr)$dump, ""))
}
