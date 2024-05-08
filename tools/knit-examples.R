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
  knit_vignette(f, external = TRUE)
}
