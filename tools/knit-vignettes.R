#!/usr/bin/env Rscript

envir::attach_source("tools/knit.R")

for (f in list.files("vignettes-src",
                     pattern = "^[^_.].+\\.[RrQq]md",
                     full.names = TRUE)) {
  knit_vignette(f)
}

