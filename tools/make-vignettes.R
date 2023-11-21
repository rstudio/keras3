library(envir)
attach_source("tools/utils.R")
# attach_source("tools/common.R")


fetch_tutobook_filepaths <- function(...) {
  c(...) %>%
    lapply(list.files, full.names = TRUE,
           recursive = TRUE, pattern = "\\.py$") %>%
    unlist() %>%
    .[!str_detect(., "/keras_(cv|nlp|tuner)/")] %>%
    .[!duplicated(basename(.))]
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
if(FALSE) {
  # one time: add tether fields
tibble(local_tutobook_path = guides) %>%
  mutate(
    name = basename(local_tutobook_path) |> fs::path_ext_remove(),
    rmd_path = glue("vignettes-src/{name}.Rmd"),
    rmd_exists = file.exists(rmd_path),

    tether_url =
      sub(
        path.expand("~/github/keras-team/keras/"),
        "https://raw.githubusercontent.com/keras-team/keras/master/",
        local_tutobook_path,
        fixed = TRUE
      ) %>%
      sub(
        path.expand("~/github/keras-team/keras-io/"),
        "https://raw.githubusercontent.com/keras-team/keras-io/master/",
        .,
        fixed = TRUE
      ) %>% fs::path()
  ) %>%
  rowwise() %>%
  mutate(write_tether_filed = {
    lines <- readLines(rmd_path) |> trimws("right")
    fm_end <- which(lines == "---")[2]
    lines[fm_end] <- sprintf("tether: %s\n---", tether_url)
    writeLines(lines, rmd_path)
  })
  relocate(tether_url) %>%
  print(n = Inf)
}

examples <-
  fetch_tutobook_filepaths %(% {
    "~/github/keras-team/keras/examples/"
    # "~/github/keras-team/keras-io/examples/"
  }


f <- "~/github/rstudio/keras/vignettes-src/writing_your_own_callbacks.Rmd"
list.files(
    c("vignettes-src", "vignettes"),
    pattern = "\\.[qQrR]?md$",
    recursive = TRUE,
    full.names = TRUE,
    all.files = TRUE
  )


old_tether <- readLines(tether_file)



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

