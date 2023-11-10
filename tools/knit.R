

# TODO: move these out of the package namespace, we don't want a knitr dep on cran
knit_man_src <- function(input, ..., output_dir) {
  library(keras)
  dir <- dirname(input)
  withr::local_dir(dir)
  message("rendering: ", dir)
  keras$utils$clear_session()
  # Set knitr options to halt on errors
  #
  og_knitr_chunks <- knitr::opts_chunk$get()
  on.exit(do.call(knitr::opts_chunk$set, og_knitr_chunks), add = TRUE)
  knitr::opts_chunk$set(error = FALSE)
  # if(FALSE) {
#
#   true_figs_dir <- paste0("../../man/figures/", basename(dir))
#   fake_figs_dir <- paste0("man/figures/", basename(dir))
#
#   message('Sys.readlink("man/figures") ', Sys.readlink("man/figures"))
#   # unlink(Sys.readlink("man/figures"), recursive = TRUE, force = TRUE)
#   # unlink("man/figures", recursive = TRUE, force = TRUE)
#   unlink("man", recursive = TRUE, force = TRUE)
#   unlink(true_figs_dir, recursive = TRUE, force = TRUE)
#   dir.create(true_figs_dir, recursive = TRUE)
#   dir.create(dirname(fake_figs_dir), recursive = TRUE)
#   file.symlink(paste0("../../", true_figs_dir),
#                fake_figs_dir)
#   }
  if(FALSE) {

    true_figs_dir <- paste0("../../man/figures/")
    fake_figs_dir <- paste0("man/figures/")

    unlink(Sys.glob(paste0("../../man/figures/", basename(dir), "-*")))
    # message('Sys.readlink("man/figures") ', Sys.readlink("man/figures"))
    # unlink(Sys.readlink("man/figures"), recursive = TRUE, force = TRUE)
    # unlink("man/figures", recursive = TRUE, force = TRUE)
    unlink("man", recursive = TRUE, force = TRUE)
    unlink("figure", recursive = TRUE, force = TRUE)
    # unlink(true_figs_dir, recursive = TRUE, force = TRUE)
    dir.create(true_figs_dir, recursive = TRUE, showWarnings = FALSE)
    dir.create(dirname(fake_figs_dir), recursive = TRUE)
    fs::link_create(
      paste0("../", true_figs_dir),
      fake_figs_dir
    )
    # file.symlink(paste0("../", true_figs_dir),
    #              fake_figs_dir)
  }
  system("ls -alR")
  # normalizePath(fake_figs_dir)
if(FALSE) {

    true_figs_dir <- paste0("../../man/figures/")
    fake_figs_dir <- paste0("figures/")


    unlink(Sys.glob(paste0("../../man/figures/", basename(dir), "-*")))
    # message('Sys.readlink("man/figures") ', Sys.readlink("man/figures"))
    # unlink(Sys.readlink("man/figures"), recursive = TRUE, force = TRUE)
    # unlink("man/figures", recursive = TRUE, force = TRUE)
    unlink("man", recursive = TRUE, force = TRUE)
    unlink("figures", recursive = TRUE, force = TRUE)
    # unlink(true_figs_dir, recursive = TRUE, force = TRUE)
    # dir.create(true_figs_dir, recursive = TRUE, showWarnings = FALSE)
    # dir.create(dirname(fake_figs_dir), recursive = TRUE)
    fs::link_create( true_figs_dir, "figures")
    # file.symlink(paste0("../", true_figs_dir),
    #              fake_figs_dir)
  # }

    knitr::opts_chunk$set(
      fig.path = paste0("figures/", basename(dir), "-"),
      # fig.width = 3, fig.height = 3, dev = "png"
    )
}
  fig.path <- paste0(basename(dir), "-")

  knitr::opts_chunk$set(
    fig.path = fig.path,
      # fig.width = 3, fig.height = 3,
    dev = "svg"
    # fig.width = 4, fig.height = 4, dev = "svg"
  )

  unlink(Sys.glob(paste0(fig.path, "*.svg")))

  knitr::knit("2-translated.Rmd", "3-rendered.md",
              quiet = TRUE, envir = new.env(parent = globalenv()))

  # browser()
  unlink(Sys.glob(paste0("../../man/figures/", fig.path, "*.svg")))
  figs <- Sys.glob(paste0(fig.path, "*.svg"))

  # link_create(
  link_path <- fs::path("../../man/figures", basename(figs))
  link_target <- fs::path_rel(figs, dirname(link_path))
  fs::link_create(link_target, link_path)
  # fs::path_rel(figs, "../../man/figures")
  # fs::path("../../man/figures", figs)
    # )
  # browser()
  # if(!length(Sys.glob(paste0("figures/", basename(dir), "-*")))) {
  #   # unlink(true_figs_dir)
  #   unlink(fake_figs_dir)
  # }

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

  # x <- sub("](figures/", "](", x, fixed = TRUE)
  # x <- sub("](man/figures/", "](", x, fixed = TRUE)

  writeLines(x, "3-rendered.md")

  message("Done!    file.edit('", file.path(dir, "3-rendered.md"), "')")

}

environment()
