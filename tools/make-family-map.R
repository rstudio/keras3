

if(!"source:tools/utils.R" %in% search()) envir::attach_source("tools/utils.R")
# remotes::install_github("hrbrmstr/pluralize")
envir::import_from(pluralize, singularize)
tryCatch(pluralize:::.onAttach(), error = identity)

make_families <- function(
    endpoint,
    module = endpoint |> paste0(".__module__") |> py_eval(),
    name = endpoint |> str_extract("[^.]+$"),
    endpoint_sans_name = endpoint |> str_extract("^keras\\.(src\\.)?(.+)\\.[^.]+$", group = 2) %|% '',
    ...) {
# browser()
  # if (module |> str_detect("schedules"))
  # if(module |> startsWith())

  chunk <- function(string) {

    str_split_1(string, fixed(".")) %>%
      setdiff(c("keras", "src")) %>%
      {
        map_chr(seq_along(.), \(i) {
          x <- .[1:i]
          # browser()
          # if(any(grepl("random_initializer", x))) browser()
          x <- x |>
            rev() |>
            str_replace_all("_", " ") |>
            str_split(" ") |>
            unlist() |>
            unique()
          # if(any(grepl("learning", x)))browser()
          if (length(x) > 1)
            x <- x[!duplicated(singularize(x), fromLast = TRUE)]

          if ((lenx <- length(x)) > 1)
            x[-lenx] <- singularize(x[-lenx])

            # x[-skip] <- x[-skip] |>
            #   singularize()
              # str_replace("e?s$", "")
          # }
          str_flatten(unique(x), " ")
        })
      }
  }

  family <- unlist(c(chunk(endpoint_sans_name),
                     chunk(module))) %>%
    setdiff(c("numpy")) %>%
    remap_families() %>%
    unique() %>%
    rev() # most precise family sorted first.

  if (endpoint |> startsWith('keras.optimizers.schedules.'))
    family %<>% setdiff("optimizers")

  if (endpoint == "keras.utils.get_source_inputs")
    family %<>% setdiff("ops")

  if (endpoint |> startsWith("keras.utils."))
    family %<>% c("utils")

  if ("image ops" %in% family)
    append(family, after = 1) <- "image utils"

  # browser()
  if (any(str_detect(endpoint_sans_name, c("data_adapters", "data_adapter_utils"))))
    family %<>% c("dataset utils", .)

  if (name |> str_detect("_dataset_from_"))
    family %<>% c("dataset utils", .)

  if (endpoint == "keras.layers.TFSMLayer")
    family %<>% c("saving", "layers", .)

  unique(family)
}; make_families('keras.layers.InputLayer')
# }; make_families('keras.optimizers.schedules.CosineDecay')
#make_families('keras.initializers.glorot_normal')
# pmap(df, make_families)

# stop()
endpoints <- list_endpoints(skip = c(
  # to be processed / done
  # "keras.saving",
  # "keras.backend",
  # "keras.dtensor",
  # "keras.mixed_precision",
  # "keras.models",
  # "keras.export",
  # "keras.experimental",
  # "keras.applications",
  # "keras.distribution",  # multi-host multi-device training
  # "keras.protobuf", # ???
  #
  # "keras.datasets",            # datasets unchanged, no need to autogen
  # "keras.preprocessing.text",  # deprecated

  "keras.estimator",           # deprecated
  "keras.legacy",
  "keras.optimizers.legacy",

  "keras.src"                  # internal
)) |> unique()


df <- tibble(
  endpoint = endpoints,
  py_obj = endpoint |> map(py_eval),
  module = py_obj |> map_chr(\(o) o$`__module__` %||% ""),
  endpoint_sans_name = endpoint |> str_extract("^keras\\.(src\\.)?(.+)\\.[^.]+$", group = 2) %|% "",
  name = endpoint |> str_extract("[^.]+$"),
  r_name = endpoint |> map_chr(make_r_name)
)


df <- df |>
  filter(!str_detect(endpoint, "keras.applications.*.(decode_predictions|preprocess_input)"))


df$family <- pmap(df, make_families)

df %<>% tidyr::unchop(family, keep_empty = TRUE)

currently_exported_r_names <- list.files("man-src", pattern = "\\.Rmd$") %>%
  fs::path_ext_remove()

# df <- df %>%

# pick the families we'll keep
keeper_families <- df %>%
  filter(r_name %in% currently_exported_r_names) %>%
  group_by(family) %>%
  summarise(
    r_names = list(r_name),
    endpoints = list(endpoint),
    n = n()
  ) %>%
  arrange(n) %>%
  select(n, family, r_names, endpoints) %>%
  filter(n >= 2 &
           !str_detect(family, '(pooling|padding|sampling|conv|cropping|lstm)[123]d')) %>%
  pull(family)


# inform which families we're omitting
cat("Dropping families: (r symbols)"); df %>%
  filter(!family %in% keeper_families) %>%
  group_by(family) %>%
  mutate(n = n()) %>%
  ungroup() %>%
  filter(n != 1) %>%  # groups of 1 don't make sense ever
  split(.$n) %>%
  lapply(\(df) {
    split(df, df$family) |>
      lapply(\(df) as.list(df$r_name))
  }) %>%
  unname() %>%
  unlist(recursive = FALSE) %>%
  as.list() %>%
  iwalk(\(r_names, family) {
    cat(sprintf("'%s': %s\n", family, str_flatten_comma(unlist(r_names))))
  })

df <- df %>%
  filter(family %in% keeper_families & family != '')

# TODO: add family tags
# - 'image utils' to k_image_*
# - `pack_x_y_sample_weight` and `unpack_x_y_sample_weight` to 'dataset utils'
# - r_name grepl("_dataset_from_") -> 'dataset utils'
# - restore 'random preprocessing layers'
# - fix dups in 'learning rate schedule optimizers'
# - remove 'metric_f1_score' single r_name family?
# - metrics/losses should be in cross-referenced.

family_to_r_names_map <- df %>%
  split(.$family) %>%
  map(\(x) x$r_name |> unlist() |> unique())

r_name_to_family_map <- df %>%
  split(.$r_name) %>%
  map(\(x) x$family |> unlist() |> unique())

dump(c("r_name_to_family_map", "family_to_r_names_map"), "tools/family-maps.R")
file.edit("tools/family-maps.R")
# after generating them, it's helpful to have RStudio reformat it w/ cmd+A, cmd+shift+a cmd+s
stop("DONE!  cmd+a  cmd+shift+a  cmd+s")

dump("r_names_and_fams", "r_names_and_fams.R")




families_df <- df %>%
  group_by(family) %>%
  summarise(
    r_names = list(r_name),
    endpoints = list(endpoint),
    n = n()
  ) %>%
  arrange(n) %>%
  select(n, family, r_names, endpoints)


r_names_and_fams <-
  tibble::enframe(fams_and_r_names, "r_name", "family") %>%
  tidyr::unchop("family") %>%
  split(.$family) %>%
  map(\(x) unique(unlist(x$r_name)))


dump("fams_and_r_names", "fams_and_r_names.R")
dump("r_names_and_fams", "r_names_and_fams.R")
jsonlite::write_json(r_names_and_fams, "r_names_and_fams.json", pretty = TRUE)
styler::style_file("r_names_and_fams.R")
styler::style_file("fams_and_r_names.R")
yaml::as.yaml(fams_and_r_names) |> cat()
yaml::as.yaml(r_names_and_fams) |> cat()

df %>%
  tidyr::unchop(family, keep_empty = TRUE) %>%
  left_join(families_df, by = join_by(family))


families_df %>%
  filter(n >= 4)
  mutate(across(c(r_names, endpoints), \(l) map_chr(l, str_flatten_comma))) %>%
  # select(-r_names) %>%
  print(n = Inf)




stop()



df_families <- df %>%
  rowwise() %>% mutate(family = list(tags$family)) %>% ungroup() %>%
  select(endpoint, r_name, family) %>%
  tidyr::unchop(family, keep_empty = TRUE)

## Inspect
symbol_families <- df_families %>%
  group_by(endpoint, r_name) %>%
  summarise(
    n = n(),
    families_pretty = family |> single_quote() |> rev() |> str_flatten_comma(),
    families = list(family)
  ) %>%
  arrange(families_pretty) |>
  print(n = Inf)

family_symbols <- df_families %>%
  group_by(family) %>%
  summarise(
    n = n(),
    r_names_pretty = r_name |> single_quote() |> rev() |> str_flatten_comma(),
    r_names = list(r_name)
  ) %>%
  arrange(n, family) |>
  print(n = Inf)


min_size_for_family <- 4
exception_families <- "io utils"

keeper_famalies <- family_symbols %>%
  filter(n >= min_size_for_family | family %in% exception_families) %>%
  pull(family)

dput_cb(keeper_famalies)
stop("Update tools/utils.R $ keeper_families")
