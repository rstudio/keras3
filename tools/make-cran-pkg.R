
devtools::build(manual = TRUE)

withr::with_dir("..", {
  system("R CMD check --as-cran keras3_*.tar.gz")
})

r"(

tools/make-man.R && \
tools/knit-vignettes.R && \
tools/knit-examples.R && \
tools/make-website.R


tools/make-man.R && tools/make-website.R

)"
