library(envir)
attach_source("tools/setup.R")
attach_source("tools/common.R")

guides <- c %(% {
  "~/github/keras-team/keras/guides"
  # "~/github/keras-team/keras/examples"
  "~/github/keras-team/keras-io/guides/keras_core"
  # "~/github/keras-team/keras-io/guides"
} %>% lapply(list.files, full.names = TRUE, recursive = TRUE) %>%
  unlist() %>%
  .[!duplicated(basename(.))]


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
        frontmatter <- yaml::as.yaml(x) %>% str_trim("right")
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

tutobook_to_rmd <- function(path_to_tutobook) {
  tutobook <- readLines(path_to_tutobook)
  name <- path_to_tutobook %>%
    basename() %>% fs::path_ext_remove() %>%
    stringr::str_to_title()

  vignette_header <- glue::glue(title = name)
  tutobook <- munge_tutobook(tutobook)

  # tutobook <<- tutobook
#
#   tutobook <- tutobook[-1] # drop opening '"""'
#   tutobook %<>% .[-(which.max(. == '"""'))]
#   tutobook %<>% .[-(which.max(. == '"""'))]
#
#   tutobook <- stringr::str_replace_all(tutobook, '^"""', "\n```")
  fs::dir_create("autogen-vignettes")
  new_path <- path_to_tutobook %>% basename() %>%
    fs::path_ext_set(".Rmd")
  writeLines(tutobook, file.path("autogen-vignettes", new_path))
}

# lapply(guides[9], tutobook_to_rmd)
lapply(guides, tutobook_to_rmd)
