

if(!"source:tools/utils.R" %in% search()) envir::attach_source("tools/utils.R")

if(!file.exists("~/github/keras-team/keras-io/scripts/master.py")) {
  python <- virtualenv_python("r-keras")
  system2(python, "-m pip install git+https://github.com/keras-team/keras-cv.git")
  system2(python, "-m pip install git+https://github.com/keras-team/keras-nlp.git")
  fs::dir_create("~/github/keras-team/")
  git("clone --depth 1 --branch master https://github.com/keras-team/keras-cv.git",
      shQuote(normalizePath("~/github/keras-team/")))
}

make_map_of_endpoints_to_keras_io_urls <- function() {

  # withr::with_dir("~/github/keras-team/keras-io", system("git pull"))
  master <- import_from_path("master",
                             "~/github/keras-team/keras-io/scripts/")$MASTER

  get_type <- function(object_) {
    if (inspect$isclass(object_)) {
      return("class")
    } else if (py_ismethod(object_)) {
      return("method")
    } else if (inspect$isfunction(object_)) {
      return("function")
    } else if (py_has_attr(object_, "fget")) {
      return("property")
    } else {
      stop(sprintf("%s is detected as not a class, a method, a property, nor a function.", object_))
    }
  }

  recursive_make_map <- function(entry, current_url) {
    current_url <- path(current_url, entry$path)
    entry_map <- list()
    if ("generate" %in% names(entry)) {
      for (symbol in entry$generate) {
        symbol <- sub("keras_core", "keras", symbol)
        object_ <- try(py_eval(symbol), silent = TRUE)
        if(inherits(object_, "try-error")) {
          next
        }
        object_type <- get_type(object_)
        object_name <- last(str_split_1(symbol, fixed(".")))

        if (startsWith(symbol, "tensorflow.keras."))
          symbol <- sub("tensorflow.keras.", "keras.", symbol, fixed = TRUE)

        object_name <- tolower(gsub("_", "", object_name))
        entry_map[[symbol]] <- paste0(current_url, "#", object_name, "-", object_type)
      }
    }

    if ("children" %in% names(entry)) {
      for (child in entry$children) {
        child_map <- recursive_make_map(child, current_url)
        entry_map <- c(entry_map, child_map)
      }
    }
    return(entry_map)
  }

  out <- recursive_make_map(master, fs::path(""))
  nms <- names(out)
  out <- map_chr(out, identity)
  # out <- sub("keras_core", "keras", out, fixed = TRUE)
  out <- path("https://keras.io/", out)
  out <- as.list(out)
  names(out) <- nms
  out
}


upstream_keras_io_urls_map <- make_map_of_endpoints_to_keras_io_urls()

dump("upstream_keras_io_urls_map",
     "tools/endpoints-to-keras-io-urls.R")


