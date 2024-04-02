#!source envir::attach_source(.file)


knit_keras_init <- function(backend = NULL) {
  if(!is.null(backend))
    keras3::use_backend(backend)
  # reticulate::use_virtualenv("r-keras")
  options(width = 76)
  keras3::clear_session()
  keras3::set_random_seed(1)
}

yaml.load <- getExportedValue("yaml", "yaml.load")
as.yaml <- getExportedValue("yaml", "as.yaml")

knit_man <- function(input, ..., output_dir) {

  cli::cli_alert('knit_man_src("{.file {input}}")')
  # message("knit_man_src(", glue::double_quote(input), ")")
  library(keras3)
  input <- normalizePath(input)
  dir <- dirname(input)
  withr::local_dir(dir)
  input <- basename(input)

  fig.path <- paste0(basename(dir), "-")
  unlink(Sys.glob(paste0("../../man/figures/", fig.path, "*.svg")))
  unlink(Sys.glob(paste0("../../man/figures/", fig.path, "*.png")))
  # unlink(Sys.glob(paste0(fig.path, "*.svg")))
  unlink(Sys.glob("*.svg"))

  og_knitr_chunks <- knitr::opts_chunk$get()
  on.exit(do.call(knitr::opts_chunk$set, og_knitr_chunks), add = TRUE)

  knitr::render_markdown()
  knitr::opts_chunk$set(
    # error = FALSE,
    # fig.path = fig.path,
    fig.width = 7, fig.height = 7,
    dev = "svg"
  )

  # og_output_hook <- knitr::knit_hooks$get("output")
  # output <- function(x, options) {
  #   x <- .keras_knit_process_chunk_output(x, options)
  #   og_output_hook(x, options)
  # }

  output <- input |> fs::path_ext_set("md")

  # knitr::render_markdown()
  # on.exit(knitr::knit_hooks$restore())

  knit_keras_init()

  knitr::knit(input, output, quiet = TRUE,
              envir = new.env(parent = globalenv()))

  # figs <- Sys.glob(paste0(fig.path, "*.svg"))
  figs <- Sys.glob("*.svg")

  if (length(figs)) {
    link_path <- fs::path("../../man/figures", basename(figs))
    link_target <- fs::path_rel(figs, dirname(link_path))
    fs::link_create(link_target, link_path)
    message("creating link ", link_path, " -> ", link_target)
  }

  # x <- readLines("3-rendered.md")
  x <- readLines(output)
  x <- trimws(x, "right")

  if(x[1] == "---") {
    stopifnot(x[3] == "---")
    x <- x[-(1:3)]
    while(x[1] == "") x <- x[-1]
  }

  # x <- process_chunk_output(x)

  writeLines(x, output, useBytes = TRUE)

}


knit_keras_process_chunk_output <- function(x, options) {
  # this hook get called with each chunk output.
  # x is a single string of collapsed lines, terminated with a final \n
  final_new_line <- endsWith(x[length(x)], "\n")
  x <- x |> strsplit("\n") |> unlist() |> trimws("right")

  # strip object addresses; no noisy diff
  x <- sub(" at 0[xX][0-9A-Fa-f]{9,16}>$", ">", x, perl = TRUE)

  # remove reticulate hint from exceptions
  x <- x[!grepl(r"{## .*rstudio:run:reticulate::py_last_error\(\).*}", x)]
  x <- x[!grepl(r"{## .*reticulate::py_last_error\(\).*}", x)]

  x <- paste0(x, collapse = "\n")
  if(final_new_line && !endsWith(x, "\n"))
    x <- paste0(x, "\n")
  x
}


knit_vignette <- function(input, ..., output_dir) {

  input.Rmd <- fs::path_real(input)
  pkg_dir <- strsplit(input.Rmd, "/vignettes-src/", fixed = TRUE)[[1]][[1]]

  if(getwd() != dirname(input.Rmd)) {
    render_dir <- dirname(input.Rmd)
    message("Changing wd to ", render_dir)
    owd <- setwd(render_dir)
    on.exit({
      setwd(owd)
    })
  }

  input.Rmd <- basename(input.Rmd)
  output.md <- sub("\\.[qrR]md$", ".md", input.Rmd)
  name <- fs::path_ext_remove(input.Rmd)

  fig.path <- name

  unlink(fig.path, recursive = TRUE)
  message("kniting: ", output.md)

  og_hooks <- knitr::knit_hooks$get()
  on.exit(knitr::knit_hooks$set(og_hooks), add = TRUE)

  knitr::render_markdown()
  og_output_hook <- knitr::knit_hooks$get("output")
  knitr::knit_hooks$set(output = function(x, options) {
    x <- knit_keras_process_chunk_output(x, options)
    og_output_hook(x, options)
  })

  knitr::opts_chunk$set(
    error = FALSE,
    fig.path = paste0(fig.path, "/")
  )

  fm <- yaml_front_matter(input.Rmd)

  knit_keras_init(fm$backend)

  withr::with_options(c(cli.num_colors = 256L), {
    knitr::knit(input.Rmd, output.md,
                envir = new.env(parent = globalenv()))
  })

  lines <- readLines(output.md)
  unlink(output.md)

  # fix frontmatter
  end_fm_i <- which(lines == "---")[2]
  x_fm <- lines[2:(end_fm_i-1)]
  fm <- yaml.load(x_fm)

  fm$knit <- NULL
  fm$output <- "rmarkdown::html_vignette"
  fm$accelerator <- NULL
  fm$tether <- NULL
  fm$author <- NULL # commented till pkgdown rendering fix

  withr::with_dir(pkg_dir, {
    stopifnot(dir.exists(".git"))
    last_modified_date <-
      # reticulate:::system2t("git", c(
      system2("git", c(
        "log -1 --pretty=format:'%ad'",
        "--date=format:'%Y-%m-%d'",
        "--", shQuote(input.Rmd)),
        stdout = TRUE
      )
  })
  # message("Last modified: ", last_modified_date)
  fm$date <- sprintf("Last Modified: %s; Last Rendered: %s",
                     last_modified_date, format(Sys.Date()))
  if(!length(fm$date))
    fm$date <- NULL
  # fm$date <- format(Sys.Date())
  vignette <- glue::glue_data(
    list(title = fm$title),
    .open = "<<", .close = ">>", r"---(
    vignette: >
      %\VignetteIndexEntry{<<title>>}
      %\VignetteEngine{knitr::rmarkdown}
      %\VignetteEncoding{UTF-8}
      )---")

  # dumping vignette via as.yaml breaks downstream, the rd entry needs to be a block
  fm <- as.yaml(fm) # has a trailing \n
  fm <- paste0(fm, vignette)

  lines <- c("---", fm, "---", lines[-(1:end_fm_i)])
  lines <- paste0(lines, collapse = "\n") |> strsplit("\n") |> _[[1L]] |> trimws("right")

  output.Rmd <- sub("/vignettes-src/", "/vignettes/", fs::path_real(input.Rmd),
                    fixed = TRUE)
  message("postprocessed output file: ", output.Rmd)
  writeLines(lines, output.Rmd, useBytes = TRUE)

  # figures dir
  if(dir.exists(fig.path)) {
    new_path <- sub("/vignettes-src/", "/vignettes/",
                    as.character(fs::path_real(fig.path)))
    unlink(new_path, recursive = TRUE)
    fs::file_move(fig.path, new_path)
  }
}

if(!interactive())
evalq({
  .Last <- function() { message("Finished!") }
}, .GlobalEnv)


# o_knitr.graphics.rel_path <- options(knitr.graphics.rel_path = FALSE)
# on.exit(options(o_knitr.graphics.rel_path), add = TRUE)


yaml_front_matter <- function(infile, lines = readLines(infile)) {
  end_fm_i <- which(lines == "---")[2]
  x_fm <- lines[2:(end_fm_i-1)]
  fm <- yaml.load(x_fm)
  fm
}

# update absolute figure links so they're relative links to the vignette dir
# lines <- sub(paste0("](", dirname(fig.path), "/"), "](", lines, fixed = TRUE) # md formatting
## could probably move this a wrapped plot hook: knit_hooks$get('plot')

# # html output, as generated by knitr::include_graphics()
# lines <- sub(paste0('src="', dirname(fig.path), "/"), 'src="', lines, fixed = TRUE)


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
