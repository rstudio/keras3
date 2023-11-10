library(envir)
attach_source("tools/utils.R")
# attach_source("tools/common.R")


munge_tutobook <- function(tutobook) {

  # browser()

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
        x$knit <- '({source("../../tools/knit.R"); knit_vignette)'
        # # x$repo <- https://github.com/rstudio/keras

        frontmatter <- yaml::as.yaml(x) |> str_trim("right")
        out <- str_flatten_lines("---", frontmatter, "---")
        return(out)
      }

      out <- .x$line |>
        str_trim("right") |>
        str_flatten_lines() |>
        str_trim()

      if(out == "")
        return("")

      if(.grp$is_code) {

        type <- .grp$section_type
        if(is.na(type) || type == "")
          type <- "python"
        out <- str_flatten_lines(sprintf("```%s", type), out, "```")

      } else {

        out <- str_compact_newlines(out)

      }
      out

    }) %>%
    keep(., . != "") %>%
    str_flatten("\n\n")

}

str_compact_newlines <- function(x, max_consecutive_new_lines = 2) {
  x <- x |> str_flatten_lines()
  while (nchar(x) != nchar(x <- gsub(
    strrep("\n", max_consecutive_new_lines + 1),
    strrep("\n", max_consecutive_new_lines),
    x, fixed = TRUE))) {}
  x
}


fetch_tutobook_filepaths <- function(...) {
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


make_guide <- function(guide) {
  # guide == path to tutobook from upstream
  name <- guide |> path_file() |> path_ext_remove()
  dir <- dir_create("vignettes-src", name)

  file_copy(guide, path("vignettes-src", name, "0-tutobook.py"), overwrite = TRUE)
  formatted_path <- path("vignettes-src", name, "1-formatted.md")
  tutobook_to_rmd(guide, outfile = formatted_path)
  translated_path <- path("vignettes-src", name, "2-translated.Rmd")
  if(!file_exists(translated_path))
    file_copy(formatted_path, translated_path)
  link <- path("vignettes-src", name, ext = "Rmd")
  if(!file_exists(link))
    link_create(path(name, "2-translated.Rmd"), link)
}

vignettes_src_pull_upstream_updates <- function() {
  dir_ls("vignettes-src/", type = "directory") |>
    walk(\(dir) {
      # withr::local_dir(dir)
      name <- path_file(dir)
      upstream_filepath <- guides %>% .[path_ext_remove(path_file(.)) == name]
      stopifnot(length(upstream_filepath) == 1)
      # browser()
      old_upstream <- read_file(dir / "0-tutobook.py")
      new_upstream <- read_file(upstream_filepath)
      if(old_upstream == new_upstream) return()

      if (file.exists(dir / "2-translated.Rmd"))
        git(
          "diff -U1 --no-index",
          "--diff-algorithm=minimal",
          paste0("--output=", dir / "translate.patch"),
          dir / "1-formatted.md",
          dir / "2-translated.Rmd",
          valid_exit_codes = c(0L, 1L)
        )

      write_lines(new_upstream, dir/"0-tutobook.py")
      rmd <- munge_tutobook(str_split_lines(new_upstream))
      # export <- mk_export(endpoint)
      write_lines(rmd, dir/"1-formatted.md")
      write_lines(rmd, dir/"2-translated.Rmd")

      if (!file.exists(dir / "translate.patch") ||
          !length(patch <- read_lines(dir / "translate.patch")))
        return()

      patch[c(1L, 3L)] %<>% str_replace(fixed("/1-formatted.md"), "/2-translated.Rmd")
      # patch <- patch[-2] # drop index <hash>..<hash> line
      write_lines(patch, dir / "translate.patch")

      git("add", dir/"2-translated.Rmd")
      git("apply --3way --recount --allow-empty", dir/"translate.patch",
          valid_exit_codes = c(0L, 1L))

    })
}

include("tools/knit.R")

vignette_src_render_translated <-
  function(directories = dir_ls("vignettes-src/", type = "directory")) {
    directories |>
      as_fs_path() |>
      # set_names(basename) %>%
      purrr::walk(\(dir) {
        # withr::local_dir(dir)
        message("rendering: ", dir)
        # keras$utils$clear_session()
        # TODO: This should really be a callr call
        # Set knitr options to halt on errors
        # knitr::opts_chunk$set(error = FALSE)
        knit_vignette(dir / "2-translated.Rmd")
      })
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



vignettes_src_pull_upstream_updates()

lapply(guides, make_guide)
# make_guide(guides[1])

vignette_src_render_translated()

# TODO: I should be using knitr::knit() directly, not rmarkdown::render()
# to avoid reflowing/rewrapping prose lines.

# TODO: there is an extra new line before ```{r} blocks in the translated rmd.


stop("DONE", call. = FALSE)


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

