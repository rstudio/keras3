#!/usr/bin/env Rscript

library(tfdatasets, exclude = c("shape"))
library(tensorflow, exclude = c("shape", "set_random_seed"))
library(keras3)


envir::attach_source("tools/knit.R")

for (f in list.files("vignettes-src",
                     pattern = "^[^_.].+\\.[RrQq]md",
                     full.names = TRUE)) {
  knit_vignette(f)
}

