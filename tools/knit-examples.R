#!/usr/bin/env Rscript

envir::attach_source("tools/knit.R")

library(tfdatasets, exclude = c("shape"))
library(tensorflow, exclude = c("shape", "set_random_seed"))
library(keras3)

files <- list.files("vignettes-src/examples",
                    pattern = "^[^_.].+\\.[RrQq]md",
                    recursive = TRUE,
                    full.names = TRUE)

# render index last
files <- unique(c(files, grep("index", files, value = TRUE)), fromLast = TRUE)

for (f in files) {
  output <- sub("vignettes-src/", "vignettes/", f, fixed=TRUE)
  if (file.exists(output)) {
    age <- difftime(Sys.time(), file.info(output)$mtime, units = "days")
    if (age < 1) {
      cli::cli_inform(c("*" = "recently rendered, skipping"))
      next
    }
  }
  knit_vignette(f, external = TRUE)
}
