#!/usr/bin/env Rscript

devtools::build(manual = TRUE)

if(length(Sys.glob("../keras3_*.tar.gz")) > 1)
  stop("too many tarballs")

withr::with_dir("..", {
  system("R CMD check --as-cran keras3_*.tar.gz")
})

cat(r"(

tools/make-man.R && \
tools/knit-vignettes.R && \
tools/knit-examples.R && \
tools/make-website.R


tools/make-man.R && tools/make-website.R

)")
