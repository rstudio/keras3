

pkgdown::build_site()

css <- readLines("tools/docs/pkgdown.css")
cat(css, file = "docs/pkgdown.css", sep = "\n", append = TRUE)

js <- readLines("tools/docs/pkgdown.js")
cat(js, file = "docs/pkgdown.js", sep = "\n", append = TRUE)

# copy to tensorflow.rstudio.com
# docs <- list.files("docs", full.names = TRUE)
# dest <- "~/websites/tensorflow.rstudio.com/_site/keras"
# unlink(dest, recursive = TRUE)
# dir.create(dest)
# file.copy(docs, dest, recursive = TRUE)
