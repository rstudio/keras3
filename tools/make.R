

if(!"source:tools/utils.R" %in% search()) envir::attach_source("tools/utils.R")
if(!"source:tools/translate-tools.R" %in% search()) envir::attach_source("tools/translate-tools.R")

# TODO: @param ... Passed on to the Python callable - scrub this from formatted.md

# TODO: revisit history - also mentions in docs (e.g., in callback_model_checkpoint())

# TODO: BackupAndRestore is broken, doesn't respect current epoch. file/fix upstream.
#       ~/github/keras-team/keras/keras/callbacks/backup_and_restore_callback.py

# TODO: swap arg order in k_vectorized_map

# TODO: reticulate, support NO_COLOR (or similar) to disable the link wrapper around `py_last_error()` hint.

# TODO: # fix `fit()` not returning `history` correctly

# TODO: add PR for purrr::rate_throttle("3 per minute")
#
# TODO: in reticulate: virtualenv_starter(): check for --enable-shared

# TODO: fix py_func(), for r_to_py.R6ClassGenerator
#   can't use __signature__ anymore in keras_core...

## TODO: "keras.applications.convnext" is a module, filtered out has good stuff

# TODO: initializer families:
# <class 'keras.initializers.constant_initializers.Zeros'>
# <class 'keras.initializers.random_initializers.RandomUniform'>

# TODO: next: losses, metrics, saving, guides/vignettes

# TODO: bidirectional, time_distributed -- need special caseing
#
# TODO: note in docs for k_logical_and (and friends) that these are dispatched
#       to from & != and so on.

# TODO: keras.Function ?? keras.Variable ?? keras.name_scope ??
#
# TODO: remove k_random_binomial() ??
#
# TODO: layer_feature_space() needs massaging.
#
# TODO: to_categorical():
#    - handle factor/character https://github.com/rstudio/keras/issues/1055
#    - make it 1 based?
#
# TODO: param descriptions - make it more robust to changes upstream
#     autoinject "see description" without needing it in the yml.
#     yml is only for explicit overrides
#
# TODO: implement and export as_shape(), make k_shape() a little nicer (e.g, an integer w/ NA)
#
# TODO: implement dim() S3 generic.
#
# TODO: remove @import methods ??
#
# TODO: add @import reticulate ??
#
# TODO: remove any tensorflow imports / DESCRIPTION deps

# The source of truth for the current translation should be...?
#    - the autogened file R/autogen-*.R, or
#    - man-src/*/2-translated.Rmd


# start by regenerating patch files

make_translation_patchfiles <- function() {
  fs::dir_ls("man-src/", type = "directory") |>
    # head(3) |>
    # keep(~str_detect(.x, "backup|_elu|_relu")) %>%
    set_names(dirname) |>
    walk(\(dir) {
      withr::with_dir(dir, {
        needs_update <- file_exists("translate.patch") &&
            file_info("2-translated.Rmd")$change_time <= file_info("translate.patch")$birth_time
        # {
        #   # message("skipping generating patch: ", dir)
        #   return()
        }
      )
      # if(!needs_update) return()
      message("updating patchfile: ", dir)
      diff <- suppressWarnings( # returns 1 on diff
        system2t("git", c("diff -U1 --no-index",
                          # "--diff-algorithm=minimal",
                          glue("--output={dir}/translate.patch"),
                      dir / "1-formatted.md", dir / "2-translated.Rmd")))
      patch_filepath <- dir/"translate.patch"
      # diff <-
        read_file(patch_filepath) |>
        str_replace_all(fixed("/1-formatted.md"), "/2-translated.Rmd") |>
        # str_replace(fixed("/1-formatted.md"), "/2-translated.Rmd") |>
        write_file(patch_filepath)

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
    }) |>
    discard(\(x) is.null(x) || x == 0) |>
    invisible()
}

# "git apply --3way --allow-empty translate.patch"

get_translations <- function() {
  fs::dir_ls("man-src/", type = "directory") |>
    set_names(basename) %>%
    keep(\(dir) read_file(path(dir, "2-translated.Rmd")) |> str_detect("```python")) |>
    head(4) |>
    purrr::walk(\(dir) {
      # browser()
      withr::local_dir(dir)
      og <- "2-translated.Rmd" |> read_file()
      new <- og |> get_translated_roxygen()
      new |> write_lines("2-translated.Rmd")
      write_rds(new, "completion.rds")
    })
}

if(FALSE) {

  get_translations()

}
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

endpoints <- list_endpoints(skip = c(
  # to be processed / done
  "keras.saving",
  "keras.backend",
  "keras.dtensor",
  "keras.mixed_precision",
  "keras.models",
  "keras.export",
  "keras.experimental",
  "keras.applications",
  "keras.legacy",
  "keras.distribution",  # multi-host multi-device training

  "keras.datasets",            # datasets unchanged, no need to autogen
  "keras.preprocessing.text",  # deprecated
  "keras.estimator",           # deprecated
  "keras.optimizers.legacy",

  "keras.src"                  # internal
)) %>%  c(
  list_endpoints("keras.applications", max_depth = 0)) %>%
  # filter out top level non module symbols for now
  grep("keras.([^.]+)$", ., value = TRUE, invert = TRUE) %>%
  unique()


# some endpoints are aliases. resolve unique endoints.
endpoints <-
  endpoints %>%
  tibble(endpoint = .) %>%
  mutate(py_obj = map(endpoint, py_eval)) %>%

  # filter non py objects, e.g., version strings
  filter(map_lgl(py_obj, inherits, "python.builtin.object")) %>%

  mutate(
    id = map_chr(py_obj, py_id),
    py_type = map_chr(py_obj, \(o) class(o) %>%
                        grep("^python\\.builtin\\.", ., value = TRUE) %>%
                        sub("python.builtin.", "", ., fixed = TRUE) %>%
                        .[1])) %>%

  # filter out literal aliases, i.e., identical py ids.
  dplyr::group_split(id) %>%
  map(\(df) {
    if(nrow(df) == 1) return(df)
    # for the pooling layer aliases, pick the longer/clearer name
    if(all(grepl(r"(keras\.layers\.(Global)?(Avg|Average|Max)Pool(ing)?[1234]D)",
                 df$endpoint)))
      return(df |> slice_max(nchar(endpoint)))

    # keep aliases of losses under metrics, for complete autocomplete with 'metric_'
    if(all(df$endpoint |> str_detect("^keras\\.(metrics|losses)\\."))) {
      # message("keeping aliases: ", str_flatten_comma(df$endpoint))
      return(df)
    }



    # if(any(df$endpoint %>% str_detect("keras.ops.average_pool"))) browser()
    return(df %>% filter(py_obj[[1]]$`_api_export_path`[[1]] == endpoint))


    # otherwise, default to picking the shortest name, but most precise
    # sort keras.preprocessing before keras.utils
    df |>
      mutate(
        name = str_split(endpoint, fixed(".")) %>% map_chr(., ~.x[length(.x)]),
        submodule = str_split(endpoint, fixed(".")) %>% map_chr(., ~.x[length(.x)-1])
      ) |>
      arrange(nchar(name), submodule) |>
      slice(1)
      # slice_min(nchar(endpoint), n = 1, with_ties = FALSE)
  }) %>%
  list_rbind() %>%

  # filter duplicate endpoint names, where all that differs is capitalization.
  # this mostly affects endpoints offered as both function and class handles:
  # metrics, losses, and merging layers.
  # if we have both functional and type api interfaces, we
  # just want the type right now (we dynamically fetch the matching
  # functional counterpart later as needed, e.g., metrics)
  split(., snakecase::to_upper_camel_case(.$endpoint)) %>%
  map(\(df) {
    if(nrow(df) == 1) return(df)
    # prefer names w/ more capital letters (i.e., the class handles)
    i <- which.max(nchar(gsub("[^A-Z]*", "", df$endpoint)))
    # message(sprintf("keep: %s drop: %s",
    #                 str_pad(df$endpoint[i], 50, "right"),
    #                 str_flatten_comma(df$endpoint[-i])))
    df[i,]
  }) %>%
  list_rbind() %>%

  .$endpoint %>% unlist() %>%

  ## filter out some endpoints that need special handling
  setdiff(c %(% {
    "keras.layers.Layer"             # only for subclassing
    "keras.optimizers.Optimizer"     # only for subclassing
    "keras.regularizers.Regularizer" # only for subclassing
    "keras.constraints.Constraint"   # only for subclassing
    "keras.initializers.Initializer" # only for subclassing
    "keras.callbacks.Callback"       # only for subclassing
    "keras.losses.Loss"              # only for subclassing
    "keras.metrics.Metric"           # only for subclassing
    "keras.optimizers.schedules.LearningRateSchedule"  # only for subclassing

    "keras.utils.PyDataset"
    "keras.utils.Sequence"           # parallel processing in R no possible this way
                                     # tfdatasets is ~100x better anyway.

    "keras.utils.plot_model"        # S3 method plot()

                                       # TODO: revisit custom_object_scope()
    "keras.utils.custom_object_scope"  # need custom work to resolve py_names -
                                       # manually wrapped to `with_custom_object_scope()`

    "keras.metrics.Accuracy"         # weird, only class handle, no fn handle - weird alias
                                     # for binary_accuracy, but without any threshold casting.
                                     # kind of confusing - the keras.metrices.<type>*_accuracy
                                     # endpoints are much preferable.

    "keras.utils.Progbar"           # needs thinking
    "keras.layers.Wrapper"           # needs thinking
    "keras.layers.InputLayer"        # use Input instead
    "keras.layers.InputSpec"         # ??
    "keras.callbacks.CallbackList"   # just an abstract list
    "keras.callbacks.History"        # always added to fit() by default

    "keras.optimizers.LegacyOptimizerWarning"
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

exports <- endpoints |>
  purrr::set_names() |>
  lapply(mk_export)

# stages:
# 1. make patchfile containing changes required from 1-formatted.md to 2-translated.md
# 2. writeout new original.md, formatted.md
# 3. apply patchfile
# 4. render translated



update_man_roxygen_dir <- function(ex) {
  message("updating ", ex$r_name)
  withr::local_dir(ex$man_roxygen_dir)
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

  ex$man_roxygen_dir <- glue("man-src/{ex$r_name}/")
  if (fs::dir_exists(ex$man_roxygen_dir))
    update_man_roxygen_dir(ex)
  else
    make_new_man_roxygen_dir(ex)
  ex
}

exports <- exports |>
  map(augment_export)


# apply_translation_patchfiles("man-src/callback_backup_and_restore")

apply_translation_patchfiles()

# stop("here")


df <- exports |>
  lapply(\(e) {
    e |>
      unclass() |>
      map_if(\(attr) ! is_scalar_atomic(attr), list) |>
      as_tibble_row()
  }) |>
  list_rbind()


df <- df |>
  mutate(endpoint_sans_name = str_extract(endpoint, "keras\\.(.*)\\.[^.]+$", 1))

df <- df |>
  arrange(endpoint_sans_name, module, r_name) |>
  mutate(file = if_else(endpoint_sans_name == "layers",
                        {
                          module |>
                            str_replace("^keras(_core)?\\.(src\\.)?", "") |>
                            str_replace(paste0(endpoint_sans_name, "\\."), "") |>
                            str_replace("^([^.]+).*", paste0(endpoint_sans_name, "-\\1.R"))
                        },
                        str_c(endpoint_sans_name %>% str_replace_all(fixed("."), "-"), ".R"))
         )

# stop("lllll")
# ?? TODO: where is ~/github/rstudio/keras/R/autogen-preprocessing.R coming from?

df |>

  group_by(file) |>
  dplyr::group_walk(\(df, grp) {

    txt <- df |>
      rowwise() |>
      mutate(final_dump = str_flatten_lines(
        # glue("# {endpoint}"),
        # glue("# {module}.{name}"),

        # str_c('r"-(', py_obj$`__doc__`, ')-"'),
        # str_c('r"-(', docstring, ')-"'),

        # str_c("#' ", readLines(fs::path(man_roxygen_dir, "2-translated.Rmd"))),
        # glue(r"--(#' @eval readLines("{fs::path(man_roxygen_dir, "2-translated.Rmd")}") )--"),

        # glue(r"--(#        file.edit("{fs::path(man_roxygen_dir, "2-translated.Rmd")}") )--"),
        glue(r"--("{fs::path(man_roxygen_dir, "1-formatted.md")}" # view the upstream doc)--"),
        glue(r"--("{fs::path(man_roxygen_dir, "2-translated.Rmd")}" # |>file.edit() # or cmd+click to edit man page)--"),

        # glue(r"--(file.edit("{fs::path(man_roxygen_dir, "2-translated.Rmd")}"))--") %>% cli::style_hyperlink(., paste0("ide:run:", .)),
        glue(r"--(#' @eval readLines("{fs::path(man_roxygen_dir, "3-rendered.md")}") )--"),
        # glue(r"--(#' @backref "{fs::path(man_roxygen_dir, "2-translated.Rmd")}" )--"),
        str_c(r_name, " <- "),
        deparse(r_fn),
        ""
      )) |>
      _$final_dump

    txt <- str_flatten(
      c("## Autogenerated. Do not modify manually.", txt),
      "\n\n\n")

    txt <- txt |>
      str_split_lines() |>
      str_remove_non_ascii() |>
      str_trim("right") |>
      str_flatten_lines() |>
      str_trim()

    txt <- txt %>% {
        while (nchar(.) != nchar(. <- gsub("#'\n#'\n", "#'\n", ., fixed = TRUE))) {} # TODO: do this in dump
        while (nchar(.) != nchar(. <- gsub("\n\n\n\n", "\n\n\n", ., fixed = TRUE))) {}
        .
      } %>%
      # handle empty @description TODO: handle this earlier, in dump.
      gsub("#' @description\n#'\n#' @", "#' @", ., fixed = TRUE)


    file <- paste0("R/autogen-", grp$file)
    # if (grp$endpoint_sans_name == "layers") {
    #   file <- "R/layers.R"
    #   unlink(file)
    # }
    # else
    #   file <- glue("R/autogen-{grp$endpoint_sans_name}.R")



    writeLines(txt, file)
  })

# devtools::document()
devtools::load_all()


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
    })
  # discard(\(x) is.null(x) || x == 0) |>
  # invisible()
}

render_roxygen_rmds()

devtools::document(roclets = c('rd', 'namespace'))
stop("DONE", call. = FALSE)



"file.edit('man-src/callback_progbar_logger/2-translated.Rmd')" %>%
  cli::style_hyperlink(., paste0("ide:run:", .)) %>%
  dput()

# file.edit('man-src/callback_progbar_logger/2-translated.Rmd')


# df <- df %>%
#   mutate(man_roxygen_dir = fs::dir_create(glue("man-src2/{r_name}/")))






ex <- exports$keras.layers.Hashing

ex2 <- augment_export(exports$keras.activations.elu)




setwd(ex2$man_roxygen_dir)
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
    mutate(write_out = withr::with_dir(man_roxygen_dir, {
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


  make_new_man_roxygen_dir <- function(ex) {
    ex$man_roxygen_dir |>
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
        library(keras)
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
