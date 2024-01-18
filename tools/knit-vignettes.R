#!/usr/bin/env Rscript

envir::attach_source("tools/knit.R")

library(tfdatasets, exclude = c("shape"))
library(tensorflow, exclude = c("shape", "set_random_seed"))
library(keras3)


files <- list.files("vignettes-src",
                    pattern = "^[^_.].+\\.[RrQq]md",
                    full.names = TRUE)


# for (f in files) {
#   cli::cli_h1(f)
#   callr::r(function(f) {
#     envir::attach_source("tools/knit.R")

#     library(tfdatasets, exclude = c("shape"))
#     library(tensorflow, exclude = c("shape", "set_random_seed"))
#     library(keras3)

#     knit_vignette(f)

#   }, args = list(f), stdout = "|", stderr = "|")
#   cat("\n")
# }


for (f in files) {
  cli::cli_h1(f)
  knit_vignette(f)
}


find_hrefs <- function(x) {
  if(!is.list(x)) return()
  unlist(c(x$href, lapply(x, find_hrefs)), use.names = FALSE)
}


listed <- yaml::read_yaml("pkgdown/_pkgdown.yml") |>
  _$navbar |>
  find_hrefs() |>
  grep("index.html", x = _, invert = TRUE, fixed = TRUE, value = TRUE) |>
  sub("articles/", "", x = _)

unlisted <- setdiff(fs::path_ext_remove(basename(files)),
                    fs::path_ext_remove(listed))
unlisted <- sprintf("vignettes-src/%s.Rmd", unlisted)

cli::cli_warn("Guides rendered but not listed in {.file pkgdown/_pkgdown.yml}")
cli::cli_ul(sprintf("{.file %s}", unlisted))
cli::cli_end()
