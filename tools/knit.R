

# TODO: move these out of the package namespace, we don't want a knitr dep on cran
knit_man_src <- function(input, ..., output_dir) {
  library(keras3)
  dir <- dirname(normalizePath(input))
  withr::local_dir(dir)
  message("knit_man_src(", glue::double_quote(input), ")")

  fig.path <- paste0(basename(dir), "-")
  unlink(Sys.glob(paste0("../../man/figures/", fig.path, "*.svg")))
  unlink(Sys.glob(paste0(fig.path, "*.svg")))

  og_knitr_chunks <- knitr::opts_chunk$get()
  on.exit(do.call(knitr::opts_chunk$set, og_knitr_chunks), add = TRUE)
  knitr::opts_chunk$set(
    error = FALSE,
    fig.path = fig.path,
    fig.width = 7, fig.height = 7,
    dev = "svg"
  )

  keras$utils$clear_session()
  set.seed(1L)
  keras$utils$set_random_seed(1L)

  knitr::knit("2-translated.Rmd", "3-rendered.md",
              quiet = TRUE, envir = new.env(parent = globalenv()))

  figs <- Sys.glob(paste0(fig.path, "*.svg"))

  if (length(figs)) {
    link_path <- fs::path("../../man/figures", basename(figs))
    link_target <- fs::path_rel(figs, dirname(link_path))
    fs::link_create(link_target, link_path)
  }

  x <- readLines("3-rendered.md")
  x <- trimws(x, "right")

  if(x[1] == "---") {
    stopifnot(x[3] == "---")
    x <- x[-(1:3)]
    while(x[1] == "") x <- x[-1]
  }

  # TODO: these filters should be confined to chunk outputs only,
  # probably as a knitr hook

  # strip object addresses; no noisy diff
  x <- sub(" at 0x[0-9A-Fa-f]{9}>$", ">", x, perl = TRUE)

  # remove reticulate hint from exceptions
  x <- x[!grepl(r"{## .*rstudio:run:reticulate::py_last_error\(\).*}", x)]
  x <- x[!grepl(r"{## .*reticulate::py_last_error\(\).*}", x)]

  writeLines(x, "3-rendered.md")

  # message("Done!    file.edit('", file.path(dir, "3-rendered.md"), "')")
}

.GlobalEnv$.Last <- function() {
  message("Finished!")
}


knit_vignette <- function(input, ..., output_dir) {
  # print(sys.call())
  # cat("wd: ", getwd(), "\n") # ~/github/rstudio/keras/vignettes-src

  library(keras3)
  # input <- normalizePath(input, mustWork = TRUE, winslash = "/") |> fs::path_tidy()
  input <- fs::path_real(input)
  output <- sub("/vignettes-src/", "/vignettes/", input, fixed = TRUE)
  output.md <- sub("\\.[qrR]md$", ".md", output)

  filename <- basename(input)
  name <- sub("\\.[qrR]md$", "", filename)

  # render_dir <- normalizePath(tempfile(paste0(name, "-")), mustWork = FALSE, winslash = "/") |> fs::path_tidy()
  render_dir <- fs::file_temp(paste0(name, "-"))
  dir.create(render_dir)
  message("Changing wd to ", render_dir)
  owd <- setwd(render_dir)
  on.exit({
    setwd(owd)
    unlink(render_dir, recursive = TRUE)
  })

  message("kniting: ", output.md)

  set.seed(1L)
  keras3:::keras$utils$clear_session()
  keras3:::keras$utils$set_random_seed(1L)

  knitr::knit(input, output.md, envir = new.env(parent = globalenv()))

  x <- readLines(output.md)
  unlink(output.md)

  end_fm_i <- which(x == "---")[2]
  x_fm <- x[2:(end_fm_i-1)]
  yaml.load <- getExportedValue("yaml", "yaml.load")
  as.yaml <- getExportedValue("yaml", "as.yaml")
  fm <- yaml.load(x_fm)

  fm$knit <- NULL
  fm$output <- "rmarkdown::html_vignette"
  fm$accelerator <- NULL
  fm$tether <- NULL
  package_dir <- strsplit(input, "/vignettes-src/", fixed = TRUE)[[1]][[1]]

  withr::with_dir(package_dir, {
    stopifnot(dir.exists(".git"))
    last_modified_date <-
      # reticulate:::system2t("git", c(
      system2("git", c(
        "log -1 --pretty=format:'%ad'",
        "--date=format:'%Y-%m-%d'",
        "--", shQuote(input)),
        stdout = TRUE
      )
  })
  # message("Last modified: ", last_modified_date)
  fm$date <- sprintf("Last Modified: %s; Last Rendered: %s",
                     last_modified_date, format(Sys.Date()))
  # TODO: fm$date <- Last compiled on `r format(Sys.time(), '%d %B, %Y')`, last updated on `r system(git `
  # fm$date <- format(Sys.Date())
  vignette <- glue::glue_data(list(title = fm$title), .trim = FALSE,
                              .open = "<<", .close = ">>",
                              "vignette: >
  %\\VignetteIndexEntry{<<title>>}
  %\\VignetteEngine{knitr::rmarkdown}
  %\\VignetteEncoding{UTF-8}")

  # dumping vignette via as.yaml breaks downstream, the rd entry needs to be a block
  fm <- as.yaml(fm) # has a trailing \n
  fm <- paste0(fm, vignette)

  x <- c("---", fm, "---", x[-(1:end_fm_i)])
  x <- paste0(x, collapse = "\n") |> strsplit("\n") |> _[[1L]] |> trimws("right")
  message("postprocessed output file: ", output)
  writeLines(x, output)
}


# TODO: move these out of the package namespace, we don't want a knitr dep on cran
# knit_man_src <- function(input, ..., output_dir) {
#   library(keras3)
#   dir <- dirname(input)
#   withr::local_dir(dir)
#   message("rendering: ", dir)
#   keras$utils$clear_session()
#   # Set knitr options to halt on errors
#   knitr::opts_chunk$set(error = FALSE)
#   file.symlink("man/figures", paste0("../../man/figures/", basename(dir)))
#   knitr::opts_chunk$set(fig.path=paste0("man/figures/", basename(dir)))
#     knitr::knit("2-translated.Rmd", "3-rendered.md",
#               quiet = TRUE, envir = new.env(parent = globalenv()))
#   x <- readLines("3-rendered.md")
#   x <- trimws(x, "right")
#   # TODO: these filters should be confined to chunk outputs only,
#   # probably as a knitr hook
#   # strip object addresses; no noisy diff
#   if(x[1] == "---") {
#     stopifnot(x[3] == "---")
#     x <- x[-(1:3)]
#     while(x[1] == "") x <- x[-1]
#   }
#   figs <- list.files("man/figures", full.names = TRUE)
#   figs_dir <- "man/figures"
#   figs_dir2 <- fs::dir_create("../../man/figures/", basename(dir))
#
#
#   file.rename(figs, new_figs_loc)
#
#   new_figs_loc <- paste0("../../man/figures/", basename(dir), basename(figs))
#   file.rename(figs, new_figs_loc)
#   file.symlink(figs, new_figs_loc)
#
#   x <- sub(" at 0x[0-9A-F]{9}>$", ">", x, perl = TRUE)
#   x <- x[!grepl(r"{## .*rstudio:run:reticulate::py_last_error\(\).*}", x)]
#   x <- x[!grepl(r"{## .*reticulate::py_last_error\(\).*}", x)]
#
#   writeLines(x, "3-rendered.md")
#
#   message("Done!    file.edit('", file.path(dir, "3-rendered.md"), "')")
#
# }



  # x <- sub("](figures/", "](", x, fixed = TRUE)
  # x <- sub("](man/figures/", "](", x, fixed = TRUE)

# figs <- list.files("man/figures", full.names = TRUE)
# figs_dir <- "man/figures"
# figs_dir2 <- fs::dir_create("../../man/figures/", basename(dir))
# # source(here::here("tools/knit.R"))$knit_man_src
#
# file.rename(figs, new_figs_loc)
#
# new_figs_loc <- paste0("../../man/figures/", basename(dir), basename(figs))
# file.rename(figs, new_figs_loc)
# file.symlink(figs, new_figs_loc)
# link_create(
# fs::path_rel(figs, "../../man/figures")
# fs::path("../../man/figures", figs)
# )
# browser()
# if(!length(Sys.glob(paste0("figures/", basename(dir), "-*")))) {
#   # unlink(true_figs_dir)
#   unlink(fake_figs_dir)
# }
# environment()

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
}
