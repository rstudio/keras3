


if(!"source:tools/utils.R" %in% search()) envir::attach_source("tools/utils.R")



tutobook_path <- "guides/distribution.py" # ~/github/keras-team/keras-io/

url <- paste0("https://raw.githubusercontent.com/keras-team/keras-io/master/",
              tutobook_path)

outfile <- sub("https://raw.githubusercontent.com/keras-team/keras-io/master/",
               "vignettes-src/", url, fixed = TRUE) |>
  fs::path_ext_set(".Rmd")


tutobook_text <- readLines(url)
tether_path <- fs::path(".tether", outfile)
fs::dir_create(dirname(tether_path))
tutobook_text |> writeLines(tether_path)
tutobook_to_rmd(url, outfile, tutobook_text)

file.edit(outfile)
