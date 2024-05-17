#!source envir::attach_source(.file)


knit_keras_init <- function(backend = NULL) {

  # reticulate::use_virtualenv("r-keras")
  if(!is.null(backend))
    keras3::use_backend(backend)
  options(width = 76)

  keras_init <- function() {
    keras3::clear_session()
    keras3::set_random_seed(1)
  }

  # this eagerly just runs the hook if keras is already loaded
  reticulate:::py_register_load_hook("keras", keras_init)
}



knit_vignette <- function(input, ..., output_dir, external = FALSE) {

  if(external) {
    return(callr::r(function(input, ...) {
      Sys.setenv(CUDA_VISIBLE_DEVICES = "0")
      options(conflicts.policy = "strict")
      conflictRules("dplyr", mask.ok = c(
        "intersect", "setdiff", "setequal", "union",
        "filter", "lag"))
      envir::import_from("tools/knit.R", knit_vignette)
      knit_vignette(input)
    }, args = list(input, ...), stdout = "", stderr = ""))
  }

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
  if(!dir.exists(output.Rmd_dir <- dirname(output.Rmd)))
    dir.create(output.Rmd_dir, recursive = TRUE)
  writeLines(lines, output.Rmd, useBytes = TRUE)

  # figures dir
  if(dir.exists(fig.path)) {
    new_path <- sub("/vignettes-src/", "/vignettes/",
                    as.character(fs::path_real(fig.path)))
    unlink(new_path, recursive = TRUE)
    fs::file_move(fig.path, new_path)
  }
}


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

if(!interactive())
evalq({
  .Last <- function() { message("Finished!") }
}, .GlobalEnv)



yaml_front_matter <- function(infile, lines = readLines(infile)) {
  end_fm_i <- which(lines == "---")[2]
  x_fm <- lines[2:(end_fm_i-1)]
  fm <- yaml.load(x_fm)
  fm
}


yaml.load <- getExportedValue("yaml", "yaml.load")
as.yaml <- getExportedValue("yaml", "as.yaml")
