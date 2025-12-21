#!/usr/bin/env Rscript


agents_path <- "AGENTS.md"
agents_tmp <- NULL
if (file.exists(agents_path)) {
  agents_tmp <- tempfile(fileext = ".md")
  file.copy(agents_path, agents_tmp, overwrite = TRUE)
  unlink(agents_path)
  on.exit({
    if (!file.exists(agents_path) && !is.null(agents_tmp) &&
        file.exists(agents_tmp)) {
      file.copy(agents_tmp, agents_path, overwrite = TRUE)
      unlink(agents_tmp)
    }
  }, add = TRUE)
}

pkgdown::build_site(devel = FALSE)

pkg <- pkgdown::as_pkgdown(".")
agents_html <- file.path(pkg$dst_path, "AGENTS.html")
agents_md <- file.path(pkg$dst_path, "AGENTS.md")
agents_dir <- file.path(pkg$dst_path, "AGENTS_files")
unlink(c(agents_html, agents_md), force = TRUE)
if (dir.exists(agents_dir)) {
  unlink(agents_dir, recursive = TRUE, force = TRUE)
}


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
