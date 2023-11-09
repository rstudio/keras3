

# TODO: move these out of the package namespace, we don't want a knitr dep on cran
knit_man_src <- function(input, ..., output_dir) {
  library(keras)
  dir <- dirname(input)
  withr::local_dir(dir)
  message("rendering: ", dir)
  keras$utils$clear_session()
  # Set knitr options to halt on errors
  knitr::opts_chunk$set(error = FALSE)
  true_figs_dir <- paste0("../../man/figures/", basename(dir))
  fake_figs_dir <- paste0("man/figures/", basename(dir))

  message('Sys.readlink("man/figures") ', Sys.readlink("man/figures"))
  # unlink(Sys.readlink("man/figures"), recursive = TRUE, force = TRUE)
  # unlink("man/figures", recursive = TRUE, force = TRUE)
  unlink("man", recursive = TRUE, force = TRUE)
  unlink(true_figs_dir, recursive = TRUE, force = TRUE)
  dir.create(true_figs_dir, recursive = TRUE)
  dir.create(dirname(fake_figs_dir), recursive = TRUE)
  file.symlink(paste0("../../", true_figs_dir),
               fake_figs_dir)
  # system("ls -al man/figures")
  # normalizePath(fake_figs_dir)


  knitr::opts_chunk$set(fig.path=paste0("man/figures/", basename(dir), "/"))
  knitr::knit("2-translated.Rmd", "3-rendered.md",
              quiet = TRUE, envir = new.env(parent = globalenv()))
  if(!length(list.files(true_figs_dir))) {
    unlink(true_figs_dir)
    unlink(fake_figs_dir)
  }
  x <- readLines("3-rendered.md")
  x <- trimws(x, "right")
  # TODO: these filters should be confined to chunk outputs only,
  # probably as a knitr hook
  # strip object addresses; no noisy diff
  if(x[1] == "---") {
    stopifnot(x[3] == "---")
    x <- x[-(1:3)]
    while(x[1] == "") x <- x[-1]
  }
  # figs <- list.files("man/figures", full.names = TRUE)
  # figs_dir <- "man/figures"
  # figs_dir2 <- fs::dir_create("../../man/figures/", basename(dir))
  # # source("../../tools/knit.R")$knit_man_src
  #
  # file.rename(figs, new_figs_loc)
  #
  # new_figs_loc <- paste0("../../man/figures/", basename(dir), basename(figs))
  # file.rename(figs, new_figs_loc)
  # file.symlink(figs, new_figs_loc)

  x <- sub(" at 0x[0-9A-F]{9}>$", ">", x, perl = TRUE)
  x <- x[!grepl(r"{## .*rstudio:run:reticulate::py_last_error\(\).*}", x)]
  x <- x[!grepl(r"{## .*reticulate::py_last_error\(\).*}", x)]

  writeLines(x, "3-rendered.md")

  message("Done!    file.edit('", file.path(dir, "3-rendered.md"), "')")

}

environment()
