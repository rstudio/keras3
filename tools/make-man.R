#!/usr/bin/env Rscript

devtools::document()

if(!"source:tools/utils.R" %in% search()) envir::attach_source("tools/utils.R")

# rd_file <- "man/activation_elu.Rd"
itemize_family <- function(rd_file) {
  x <- rd_file |> readLines() |> trimws(which = "right")

  for (i in grep("^Other .+:$", x)) {
    x[i] %<>% str_c("\n\\itemize{")

    while (startsWith(x[i <- i + 1L], "\\code{\\link{"))
      x[i] <- x[i] |> str_prefix("\\item ") |> str_remove(",$")

    x[i] %<>% str_prefix("}\n")
  }

  writeLines(x, rd_file)

}

cr_family <- function(rd_file) {
  x <- rd_file |> readLines() |> trimws(which = "right")

  for (i in grep("^Other .+:$", x)) {
    # x[i] %<>% str_c("\n\\itemize{")
    x[i] %<>% str_c(" \\cr")
    while (startsWith(x[i <- i + 1L], "\\code{\\link{"))
      x[i] %<>% str_remove(",$") %>% str_c(" \\cr")

    # x[i] %<>% str_prefix("}\n")
  }

  writeLines(x, rd_file, useBytes = TRUE)

}


Sys.glob("man/*.Rd") %>%
  walk(cr_family)
  # walk(itemize_family)
