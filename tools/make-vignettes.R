library(envir)
attach_source("tools/utils.R")
# attach_source("tools/common.R")


munge_tutobook <- function(tutobook) {

  df <- tibble(
    line = tutobook
  )

  df %>%
    mutate(
      is_delim = startsWith(line, '"""'),
      section_id = cumsum(is_delim),
      is_code = !(section_id %% 2) & !is_delim,
      delim_header = if_else(is_delim, str_replace(line, '^"""', ""), NA)) %>%
    group_by(section_id) %>%
    mutate(section_type = zoo::na.locf0(delim_header)) %>%
    ungroup() %>%
    filter(!is_delim) %>%
    group_by(section_id, is_code, section_type) %>%
    dplyr::group_map(\(.x, .grp) {

      if(.grp$section_id == 1) {
        x <- str_split_fixed(.x$line, ": ", 2)
        x[,1] %<>% snakecase::to_snake_case() %<>% str_replace_all("_", "-")
        x <- rlang::set_names(nm = x[,1], as.list(x[,2]))
        x$output <- "rmarkdown::html_vignette"
        vignette <- glue_data(list(title = x$title), r"---(
        vignette: >
          %\VignetteIndexEntry{<<title>>}
          %\VignetteEngine{knitr::rmarkdown}
          %\VignetteEncoding{UTF-8}
        )---", .open = "<<", .close = ">>")
        # x$repo <- https://github.com/rstudio/keras

        frontmatter <- yaml::as.yaml(x) %>% str_trim("right")
        frontmatter %<>% str_flatten_lines(vignette)
        frontmatter %<>% str_flatten_lines(., trim(r"---{
        knit: >
          (function(input, encoding) rmarkdown::render(
            input, encoding=encoding,
            output_file='03-rendered.md')))
        }---"))

        # frontmatter <- str_c(x[,1], ": ", x[,2])
        out <- str_flatten_lines("---", frontmatter, "---")
        return(out)
      }
        # browser()

      out <- .x$line %>%
        str_trim("right") %>%
        str_flatten_lines() %>%
        str_trim()

      if(out == "")
        return("")

      if(.grp$is_code) {

        type <- .grp$section_type
        # if(i)
        if(is.na(type) || type == "")
          # browser()
        # if(type == "")
          type <- "python"
        out <- str_flatten_lines(
          sprintf("```%s", type), out, "```")
      } else {
        out <- out %>%
          str_replace_all("\n\n\n", "\n\n") %>%
          str_replace_all("\n\n\n", "\n\n") %>%
          str_replace_all("\n\n\n", "\n\n") %>%
          str_replace_all("\n\n\n", "\n\n")
      }
      out

    }) %>%
    keep(., . != "") %>%
    str_flatten("\n\n")
  # print()
  # print(n = Inf)

}


fetch_tutobook_filepaths <- function(...) {
  # Sys.glob(c(...)) %>%
  c(...) %>%
    lapply(list.files, full.names = TRUE,
           recursive = TRUE, pattern = "\\.py$") %>%
    unlist() %>%
    .[!str_detect(., "/keras_(cv|nlp|tuner)/")] %>%
    .[!duplicated(basename(.))]

}
# tutobook = readLines(path_to_tutobook),

tutobook_to_rmd <- function(path_to_tutobook, outfile = NULL, tutobook = NULL) {
  if(is.null(tutobook))
    tutobook <- readLines(path_to_tutobook)
  if(is.null(outfile) && !missing(path_to_tutobook)) {
    outfile <- path_ext_set(path_to_tutobook, "Rmd")
  }
  # tutobook <- readr::read_file(path_to_tutobook)
  # name <- path_to_tutobook %>%
  #   basename() %>% fs::path_ext_remove() %>%
  #   stringr::str_to_title()

  # vignette_header <- glue::glue(title = name)
    tutobook <- try({
        munge_tutobook(tutobook)
    }, silent = TRUE)
  if(inherits(tutobook, "try-error")) {
    message("converting failed: ", path_to_tutobook)
    return()
  }

  # new_filename <- basename(path_to_tutobook) %>% fs::path_ext_set(".Rmd")
  # fs::dir_create(outdir)
  if(is.null(outfile))
    tutobook
  else {
    writeLines(tutobook, outfile)
    invisible(outfile)
  }

}


guides <-
  fetch_tutobook_filepaths %(% {
    "~/github/keras-team/keras/guides"
    "~/github/keras-team/keras-io/guides/keras_core"
    "~/github/keras-team/keras-io/guides"
  }

examples <-
  fetch_tutobook_filepaths %(% {
    "~/github/keras-team/keras/examples/"
    # "~/github/keras-team/keras-io/examples/"
  }


make_guide <- function(guide) {
  # guide == path to tutobook from upstream
  name <- guide |> path_file() |> path_ext_remove()
  dir <- dir_create("vignettes-src", name)
  file_copy(guide, path("vignettes-src", name, basename(guide)), overwrite = TRUE)
  file_copy(guide, path("vignettes-src", name, path_ext_set(name, "Rmd")), overwrite = TRUE)

  # file_copy(guide, path("vignettes-src", name, "0-tutobook.py"), overwrite = TRUE)
  tutobook_to_rmd(guide, outfile = path(dir, "1-formatted.md"))
  tutobook_to_rmd(guide, outfile = path(dir, "2-translated.Rmd"))
}

lapply(guides, make_guide)
# make_guide(guides[1])

stop()

# lapply(guides[9], tutobook_to_rmd)
lapply(guides, tutobook_to_rmd, outdir = "vignettes/guides")
lapply(examples, tutobook_to_rmd, outdir = "vignettes/examples")












vignette_header_template <- vignette_header <- function(title) {

  template <- r'----(
---
title: "<<title>>"
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{<<title>>}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}
---
)----'
  glue::glue(template, .open = "<<", .close = ">>")
}

