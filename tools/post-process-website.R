#!/usr/bin/env Rscript

envir::import_from(magrittr, `%<>%`)



# fix "Source: vignettes/foo.Rmd" link to point to "Source: vignettes-src/foo.Rmd"
# in head of website articles
for (file in Sys.glob(c("docs/articles/*.html",
                        "docs/articles/examples/*.html"))) {
  lines <- readLines(file)
  i <- grep("Source: ", lines, fixed = TRUE)
  lines[i] %<>% gsub("vignettes/", "vignettes-src/", ., fixed = TRUE)
  writeLines(lines, file, useBytes = TRUE)
}
