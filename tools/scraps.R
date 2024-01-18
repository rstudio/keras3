# defered TODOs:
# TODO: consider using quilt or stgit instead of the manual git calls
# https://stacked-git.github.io   https://savannah.nongnu.org/projects/quilt/ https://blog.tfnico.com/2020/07/git-tools-for-keeping-patches-on-top-of.html




# filter out function handles that also have class handles
endpoints <- endpoints %>%
  lapply(\(endpoint) {
    if(!any(startsWith(endpoint, c("keras.losses.", "keras.metrics."))))
      return(endpoint)

    py_obj <- py_eval(endpoint)
    if (!inherits(py_obj, "python.builtin.function"))
      return(endpoint)


    class_endpoint <- switch(endpoint,
                             "keras.losses.kl_divergence" = "keras.losses.KLDivergence",
                             str_replace(endpoint, py_obj$`__name__`,
                                         snakecase::to_upper_camel_case(py_obj$`__name__`)))
    tryCatch({
      py_eval(class_endpoint)
      return(NULL)
    },
    python.builtin.AttributeError = function(e) {
      # don't emit warning about known function handles without class handle
      # counterparts
      if (!endpoint %in% sprintf(
        "keras.metrics.%s", c(
          "binary_focal_crossentropy",
          'categorical_focal_crossentropy',
          'huber',
          'kl_divergence',
          'log_cosh')))
        # browser()
        print(e)
      endpoint
    })
  }) %>%
  unlist() %>%
  invisible()

get_metric_counterpart_endpoint <- function(..., fn, type) {
  if(missing(fn)) {
    endpoint <- type
    py_obj <- py_eval(endpoint)
    name <- py_obj$`__name__`
    endpoint_fn <- str_replace(endpoint, name, snakecase::to_snake_case(name))
    py_obj_fn <- py_eval(endpoint_fn) %error% browser()
  }
}



list.files("man-src", pattern = "\\.Rmd$", full.names = TRUE) %>%
  map(\(f) {
    f |>
      read_lines() |>
      str_replace_all("([^ ])=([^ ])", "\\1 = \\2") |>
      # TODO `True`, None
      str_trim("right") %>%
      write_lines(f)
  })

# {
#
#   render_roxygen_rmds <- function(filepath = fs::dir_ls("man-src/", type = "directory")) {
#     # this should probably happen in a separate R session...
#     filepath |>
#       fs::as_fs_path() |>
#       # head(3) |>
#       set_names(basename) %>%
#       purrr::walk(\(dir) {
#         withr::local_dir(dir)
#         message("rendering: ", dir)
#         keras$utils$clear_session()
#         # Set knitr options to halt on errors
#         knitr::opts_chunk$set(error = FALSE)
#         knitr::knit("2-translated.Rmd", "3-rendered.md",
#                     quiet = TRUE, envir = new.env())
#         x <- readLines("3-rendered.md")
#         # TODO: these filters should be confined to chunk outputs only,
#         # probably as a knitr hook
#         x <- x |> str_replace_all(" at 0x[0-9A-F]{9}>$", ">")
#         x <- x[!str_detect(x, r"{## .*rstudio:run:reticulate::py_last_error\(\).*}")]
#         x |> writeLines("3-rendered.md")
#       })
#     # discard(\(x) is.null(x) || x == 0) |>
#     # invisible()
#   }
# }


#
# man_src_pull_upstream_updates2 <- function() {
#
#   vscode_settings <- og_vscode_settings <-
#     jsonlite::read_json(".vscode/settings.json")
#   vscode_settings %<>% modifyList(list("git.autorefresh" = FALSE,
#                                        "git.autofetch" = FALSE))
#   jsonlite::write_json(vscode_settings, ".vscode/settings.json")
#   withr::defer(jsonlite::write_json(og_vscode_settings, ".vscode/settings.json",
#                                     pretty = TRUE))
#
#   fs::dir_ls("man-src/", type = "directory") |>
#     # head(3) |>
#     # keep(~str_detect(.x, "backup|_elu|_relu")) %>%
#     set_names(dirname) |>
#     walk(\(dir) {
#       # withr::local_dir(dir)
#       old_upstream <- read_lines(path(dir, "0-upstream.md"))
#       endpoint <- old_upstream[1]
#       old_upstream <- str_flatten_lines(old_upstream)
#       new_upstream <- format_man_src_0(endpoint)
#       # if(new_upstream == old_upstream) return() # nothing to update
#
#
#       export <- mk_export(endpoint)
#
#       if (!file.exists(dir / "2-translated.Rmd")) {
#         write_lines(export$roxygen, dir/"1-formatted.md")
#         write_lines(export$roxygen, dir/"2-translated.Rmd")
#         return()
#       }
#       write_lines(export$roxygen, dir/"1-formatted-new.md")
#       write_lines(export$roxygen, dir/"2-translated-new.Rmd")
#
#         # -m --merge <path1> <path2> <base> <result>
#         #   Perform a three-way merge by providing paths for two modified versions of
#         #   a file, the common origin of both modified versions and the output file
#         #   to save merge results.
#
#       system2("code",  c("--merge",
#               dir / "2-translated.Rmd",
#               dir / "1-formatted-new.md",
#               dir / "1-formatted.md",
#               dir/ "2-translated.Rmd"))
#       # stop()
#       # if(!identical(res, 0L)) {
#       #   cat("res <- "); dput(res)
#       #   stop("non-0 exit from git apply")
#       # }
#       # stop()
#     })
# }; man_src_pull_upstream_updates2()


# man_src_reformat_0 <- function() {}

# man_src_pull_upstream_updates()
#
#
# man_src_render
#
# vignettes_src_pull_upstream_updates
# vignettes_src_render



make_translation_patchfiles <- function() {
  fs::dir_ls("man-src/", type = "directory") |>
    # head(3) |>
    # keep(~str_detect(.x, "backup|_elu|_relu")) %>%
    set_names(dirname) |>
    walk(\(dir) {
      # withr::with_dir(dir, {
      #   # timestamps don't work because the translation file gets rewritten.
      #   doesnt_need_update <- file_exists("translate.patch") &&
      #       file_info("2-translated.Rmd")$change_time < file_info("translate.patch")$birth_time
      #   }
      # )
      # if(doesnt_need_update) return()
      # message("updating patchfile: ", dir)
      diff <- suppressWarnings( # returns 1 on diff
        system2t("git", c("diff -U1 --no-index",
                          # "--diff-algorithm=minimal",
                          glue("--output={dir}/translate.patch"),
                          dir / "1-formatted.md", dir / "2-translated.Rmd")))
      patch_filepath <- dir/"translate.patch"
      patch <- read_lines(patch_filepath)
      if(!length(patch)) return()
      # if(grepl("hard_sigmoi", dir)) browser()
      patch <- patch[-2] # drop index <hash>..<hash> line
      patch[1:3] <- str_replace(patch[1:3], fixed("/1-formatted.md"), "/2-translated.Rmd")
      write_lines(patch, patch_filepath)

      # str_replace(fixed("/1-formatted.md"), "/2-translated.Rmd") |>

      # | sed 's|1-formatted.md|2-translated.Rmd|' > translate.patch")
      # system2("git", c("diff --output=translate.patch --diff-algorithm=minimal -U1 --no-index",
      # "1-formatted.md 2-translated.Rmd "))
    })
}

apply_translation_patchfiles <- function(filepath = fs::dir_ls("man-src/", type = "directory") ) {
  filepath |>
    fs::as_fs_path() |>
    # head(3) |>
    set_names(basename) %>%
    purrr::walk(\(dir) {
      # withr::local_dir(dir)
      system2t("git", c("apply --no-index --recount --allow-empty", # --3way
                        dir/"translate.patch"))
      # system(glue("patch -p2 < {dir}/translate.patch"))
      # TODO: this needs to propogate errors up from applying the patchfile
    }) |>
    discard(\(x) is.null(x) || x == 0) |>
    invisible()
}

# "git apply --3way --allow-empty translate.patch"

make_translation_patchfiles()


# z <- "man-src/activation_elu/2-translated.Rmd" |> read_file() |>
#   get_translated_roxygen()
#
# apply_translation_patchfiles()
# write_lines(z, "man-src/activation_elu/2-translated.Rmd") -> z2


# source("tools/utils.R")


if(FALSE)
  local({
    x <- keras$callbacks$BackupAndRestore
    x$`__doc__` <- str_replace(keras$callbacks$BackupAndRestore$`__doc__`, "Note that the user", "Note that the user  asdfa asdfdsaf asdfasdf ")
  })

# fs::path("tools/raw", gsub(".", "-", endpoint, fixed = TRUE), ext = "R")
#
# make_patch_files <- function() {
#   fs::dir_ls("man-src/", type = "directory") |>
#     # head(3) |>
#     purrr::walk(\(dir) {
#   # fs::dir_map("man-src/", type = "directory", \(dir) {
#     withr::local_dir(dir)
#     # print(dir)
#     # https://git-scm.com/docs/git-diff
#     system2("git", c("diff --output=translate.patch --diff-algorithm=minimal -U1",
#                      "1-formatted.md 2-translated.Rmd"))
#   }) #|> invisible()
# }
# make_patch_files()
#
# apply_patch_files <- function() {
#   fs::dir_map("man-src/", \(dir) {
#     withr::local_dir(dir)
#
#   })
# }



update_man_src_dir <- function(ex) {
  message("updating ", ex$r_name)
  withr::local_dir(ex$man_src_dir)
  # old_docstring <- "0-docstring.md" |> readLines() |> str_flatten_lines()
  # is_changed <- ex$docstring != old_docstring
  # if(!is_changed) {
  #   message("returning early")
  #   return(ex)
  # }
  import_from({ex}, docstring, roxygen)

  # old_docstring_as_roxygen <- read_file("1-formatted.md")
  # old_roxygen_rmd <- read_file("2-translated.Rmd")
  # system("git diff -u1 1-formatted.md 2-translated.Rmd > fixup.patch")

  writeLines(docstring, "0-docstring.md")           # 0-original.md
  writeLines(roxygen, "1-formatted.md")  # 1-formatted.md
  writeLines(roxygen, "2-translated.Rmd")              # 2-translated.md
  writeLines(str_flatten_lines(                   # 3-rendered.md
    str_c(ex$r_name, " <-"),
    deparse(ex$r_fn)
  ), "function.R")

  # res <- system("git apply --3way translate.patch") # --unidiff-zero --verbose

  ex
  # if res == error: nothing else: knitr::knit() ?
  # knitting should really be a separate step...
}



augment_export <- function(ex) {

  ex$man_src_dir <- glue("man-src/{ex$r_name}/")
  if (fs::dir_exists(ex$man_src_dir))
    update_man_src_dir(ex)
  else
    make_new_man_src_dir(ex)
  ex
}

exports <- exports |>
  map(augment_export)


# apply_translation_patchfiles("man-src/callback_backup_and_restore")

apply_translation_patchfiles()

# stop("here")















render_roxygen_rmds <- function(filepath = fs::dir_ls("man-src/", type = "directory")) {
  # this should probably happen in a separate R session...
  filepath |>
    fs::as_fs_path() |>
    # head(3) |>
    set_names(basename) %>%
    purrr::walk(\(dir) {
      withr::local_dir(dir)
      message("rendering: ", dir)
      keras$utils$clear_session()
      # Set knitr options to halt on errors
      knitr::opts_chunk$set(error = FALSE)
      knitr::knit("2-translated.Rmd", "3-rendered.md",
                  quiet = TRUE, envir = new.env())
      x <- readLines("3-rendered.md")
      # TODO: these filters should be confined to chunk outputs only,
      # probably as a knitr hook
      x <- x |> str_replace_all(" at 0x[0-9A-F]{9}>$", ">")
      x <- x[!str_detect(x, r"{## .*rstudio:run:reticulate::py_last_error\(\).*}")]
      x |> writeLines("3-rendered.md")
    })
  # discard(\(x) is.null(x) || x == 0) |>
  # invisible()
}

render_roxygen_rmds()

devtools::document(roclets = c('rd', 'namespace'))
stop("DONE", call. = FALSE)
#   git add man-src/*/2-*


"file.edit('man-src/callback_progbar_logger/2-translated.Rmd')" %>%
  cli::style_hyperlink(., paste0("ide:run:", .)) %>%
  dput()

# file.edit('man-src/callback_progbar_logger/2-translated.Rmd')


# df <- df %>%
#   mutate(man_src_dir = fs::dir_create(glue("man-src2/{r_name}/")))






ex <- exports$keras.layers.Hashing

ex2 <- augment_export(exports$keras.activations.elu)




setwd(ex2$man_src_dir)
rmarkdown::render("2-translated.Rmd", #run_pandoc =
                  # output_format = "github_document",
                  output_format = rmarkdown::github_document(
                    hard_line_breaks = TRUE,
                    html_preview = FALSE
                    # pandoc_args = c()
                  ),
                  # quiet = TRUE,

                  output_file = "roxygen.md", envir = new.env())

knitr::knit("2-translated.Rmd", "roxygen.md")


setwd(here::here())



ex2 <- augment_export(export)

df %>%
  split_by


if(FALSE) {


  df %>%
    rowwise() %>%
    mutate(write_out = withr::with_dir(man_src_dir, {
      writeLines(docstring, "0-docstring.md")
      writeLines(roxygen, "roxygen_intermediate.md")
      writeLines(roxygen, "2-translated.Rmd")
      writeLines(roxygen, "roxygen.md")
    }))

  # dump_file = glue("src/wrappers/{r_name}/r_wrapper_raw.R")) %>%



  df <- dfo %>%
    mutate(dump_file = glue("src/wrappers/{r_name}/r_wrapper_raw.R")) %>%
    rowwise() %>%
    mutate(
      raw_dumpfile = str_flatten_lines(
        str_glue("# {endpoint}"),
        str_glue("# {module}.{name}"),
        str_c('r"-{', trim(docstring), '}-"'),
        "",
        dump
      ),
      old_dumpfile = dumpfile_path %>% {if(file.exists(.)) {message("exists: ", .); str_flatten_lines(readLines(.))} else "''"},
      old_docstring = str2expression(old_dumpfile)[[1]]) %>%
    ungroup() %>%
    mutate(is_new = !file.exists(dumpfile_path),
           is_changed = (!is_new) & docstring != old_docstring)

  df_new <- df %>% filter(is_new)

  df_new %>%
    rowwise() %>%
    mutate(writeout = {
      writeLines(dumpfile, dumpfile_path)
    })

  df_changed <- df %>% filter(is_changed)

  stop("!!!")




  df <- df %>%

    rowwise() %>%
    mutate(
    ) %>%

    # writeLines(txt, dump_filepath)
    # dump_filepath
    # ) %>%
    ungroup()


  df$dumpfile

}

# filter(endpoint_sans_name %in% c("layers", "ops", "constraints", "initializers",
#                                  "callbacks", "optimizers",
#                                  "preprocessing",
#                                  "preprocessing.image",
#                                  "preprocessing.sequence",
#                                  "losses", "metrics",
#                                  "optimizers.schedules",
#                                  # "utils",
#                                  "applications",
#                                  "activations", "regularizers")) |> #
# select(endpoint, r_name, module, type) |>
# |>
dplyr::bind_rows()

map(xx, format)

# tibble(endpoint = _) |>
mutate(name = endpoint |>
         str_replace_all(fixed("."), "$") |>
         map(\(s) names(eval(str2lang(s))))) |>
  tidyr::unchop(name) |>
  filter(!name %in% c("experimental", "deserialize", "serialize")) |>
  mutate(endpoint = str_c(endpoint, name, sep = ".")) |>
  filter(!endpoint %in% "keras.Sequential.build") #|>


# start w/ the export endpoint, and augment it.



x <- mk_export("keras.layers.PReLU")
x


# TODO:
#
# Parsing of params for keras$ops$tensordot isn't right
#
# TODO: @returns required for every obj
#
#
# TODO:
# layer_wrapper?
# layer_torch_module_wrapper?
# layer_time_distributed?
# layer_tfsm_layer?
# layer_bidirectional

# Seemingly removed:
# layer_random_height
# layer_random_width
# layer_locally_connected_{12}d
# layer_cudnn_*
# layer_dense_features
# layer_alpha_dropout
# layer_activation_thresholded_relu

# TODO: "activation layers" not "activations layers" as a family tag


# map_chr(df$tags, ~.x$"family" %||% '') %>% unique() %>% writeLines()
# activation functions
# callback

# core layers
# convolutional layers
# pooling layers
# activations layers
# merging layers
# normalization layers
# reshaping layers
# attention layers
# preprocessing layers
# regularization layers
# recurrent layers


# constraint
# initializer
# loss
# metric
# ops
# optimizer
# learning_rate_schedule


make_r_fn("keras.layers.PReLU")

mk_export("keras.layers.PReLU")
mk_export("keras.applications.VGG16")


mk_export("keras.ops.conv_transpose")$dump |> cat()

roxygen2::parse_file("/Users/tomasz/github/rstudio/keras/R/autogen-layers-core.R") -> r

# TODO: UpSampling1D needs fixup

r[[3]] -> r

r
r$tags[[2]]
trace(system2, quote(message(paste("+", paste0(env, collapse = " "),
                                   shQuote(command), paste0(shQuote(args), collapse = " "))))); tools:::..Rd2pdf(".")


mk_export("keras.layers.LayerNormalization")$doc$description -> d


str()

str_sppulit_1(fixed("\n```")) %>%
  {
    length(.) <- ceiling(length(.)/2)*2 # maybe resize to multiple of 2
    dim(.) <- c(2, length(.) / 2)
    rownames(.) <- c("prose", "code")
    .
  } %>% {
    p <- .["prose",]

  }
print()
str()


# TODO: callback_progbar logger ... the documented arg doesn't seem to actually be accepted,
# also, does it make sense to export it?
#
"keras.constraints.to_snake_case" # ??

# if (any(str_detect(df$endpoint, "metrics|losses"))) {
#
#   nms_in <- names(df)
#   df <- df |>
#     mutate(
#       name = map_chr(py_obj, \(o) o$`__name__`),
#       module = map_chr(py_obj, \(o) o$`__module__`)) |>
#     rowwise() %>%
#     mutate(
#       dist_from_true_home = adist(str_replace(endpoint, name, ""),
#                                   module)
#     ) |>
#     ungroup() |>
#     arrange(dist_from_true_home, desc(nchar(name)))
#     # message(str_flatten_comma(df$endpoint, ", "))
#   df %>%
#     # slice(1) %>%
#     select(!!nms_in)

# } else {

#
#
# l <- keras$layers$PReLU
#
#
# l$`__doc__` |> trim() |> split_docstring_into_sections()
#
#   cat()

n_openai_calls_made <- 0L
max_openai_calls <- 1L


make_new_man_src_dir <- function(ex) {
  ex$man_src_dir |>
    fs::dir_create() |>
    withr::local_dir()

  ex$is_new <- TRUE
  ex$is_changed <- FALSE
  ex$old_docstring <- ""
  ex$old_roxygen_intermediate <- ""

  import_from({ex}, docstring, roxygen)

  writeLines(docstring, "0-docstring.md")
  writeLines(roxygen, "1-formatted.md")

  do_call_openai <-
    (n_openai_calls_made < max_openai_calls) &&
    str_detect(roxygen, "```python")

  if (do_call_openai) {
    ex$completion <- completion <- get_translated_roxygen(roxygen)
    n_openai_calls_made <<- n_openai_calls_made + 1L
    write_rds(completion, "completion.rds")
    writeLines(completion, "2-translated.Rmd")
    tryCatch({
      library(keras3)
      keras$utils$clear_session()
      knitr::knit("2-translated.Rmd", "roxygen.md",
                  quiet = TRUE, envir = new.env())
    }, error = function(e) {
      warning("Failed to render docs for", ex$r_name)
    })
  } else {
    writeLines(roxygen, "2-translated.Rmd")
    file.copy("2-translated.Rmd", "roxygen.md")
  }

  ex
}


if(FALSE) {
  exports$keras.activations.relu$docstring <-
    read_file(fs::path("man-src",
                       exports$keras.activations.relu$r_name,
                       "0-docstring.md"))
  augment_export(exports$keras.activations.relu)


}
