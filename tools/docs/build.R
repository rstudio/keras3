

pkgdown::build_site()

css <- readLines("tools/docs/pkgdown.css")
cat(css, file = "docs/pkgdown.css", sep = "\n", append = TRUE)

js <- readLines("tools/docs/pkgdown.js")
cat(js, file = "docs/pkgdown.js", sep = "\n", append = TRUE)


source("tools/docs/build-repos.R")
