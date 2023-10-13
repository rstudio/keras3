#' Install TensorFlow and Keras, including all Python dependencies
#'
#' This function will install Tensorflow and all Keras dependencies. This is a
#' thin wrapper around [`tensorflow::install_tensorflow()`], with the only
#' difference being that this includes by default additional extra packages that
#' keras expects, and the default version of tensorflow installed by
#' `install_keras()` may at times be different from the default installed
#' `install_tensorflow()`. The default version of tensorflow installed by
#' `install_keras()` is "`r default_version`".
#'
#' @details The default additional packages are:
#' `r paste(default_extra_packages("nightly"), collapse = ", ")`, with their
#'   versions potentially constrained for compatibility with the
#'   requested tensorflow version.
#'
#' @inheritParams tensorflow::install_tensorflow
#'
#' @param tensorflow Synonym for `version`. Maintained for backwards.
#'
#' @seealso [`tensorflow::install_tensorflow()`]
#' @export
install_keras <- function(...,
                          envname = "r-keras",
                          extra_packages = NULL,
                          backend = c("tensorflow", "jax", "pytorch")
                          # # envname = "r-keras",
                          # # new_env = identical(envname, "r-keras")
                          ) {

  # if(identical(envname, "r-keras") &&
  #    reticulate::virtualenv_exists(envname))
  #   reticulate::virtualenv_remove(envname, confirm = FALSE)

  python <- envname |>
    virtualenv_create("3.10", force = identical(envname, "r-keras"), packages = NULL) |>
    virtualenv_python()

  # withr::local_dir(withr::local_tempdir())
  withr::local_dir("~/github/keras-team/keras")
  system("git pull")
  system2(python, c("-m pip install -r requirements.txt"))
  system2(python, c("pip_build.py --install"))
  message("Done!")

  return(invisible())


  pgs <- c("keras-core", extra_packages, backend[1])
  reticulate::py_install(
    c("tensorflow"),
    envname = envname,
    python_version = "3.10",
    ...
  )

  # Error presented, but install succeeds:
  # ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
  # tensorflow-macos 2.14.0 requires keras<2.15,>=2.14.0, but you have keras 3.0.0 which is incompatible.
  reticulate::py_install(
    "git+https://github.com/keras-team/keras#egg=keras",
    envname = envname,
    python_version = "3.10",
    ...
  )

}

default_version <- numeric_version("2.13")

default_extra_packages <- function(tensorflow_version = "default") {
  pkgs <- c(
    "tensorflow-hub",
    "tensorflow-datasets",
    "scipy",
    "requests",
    "pyyaml",
    "Pillow",
    "h5py",
    "pandas",
    "pydot")
  names(pkgs) <- pkgs
  v <- tensorflow_version

  if(grepl("nightly|release", v))
    return(pkgs)

  ## extract just the version
  # drop potential suffix
  v <- sub("-?(gpu|cpu)$", "", v)
  # treat rc as regular patch release
  v <- sub("rc[0-9]+", "", v)

  constraint <- sub("^([><=~]{,2}).*", "\\1", v)
  v <- substr(v, nchar(constraint)+1, nchar(v))

  if(v %in% c("default", "")) # "" might be from cpu|gpu
    v <- default_version

  v <- numeric_version(v)
  if(nzchar(constraint)) {
    # try to accommodate user supplied constraints by bumping `v` up or down
    l <- length(unclass(v)[[1]])
    switch(constraint,
           ">" = v[[1, l + 1]] <- 1,
           "<" = {
             v <- unclass(v)[[1]]
             if(v[l] == 0) l <- l-1
             v[c(l, l+1)] <- c(v[l] - 1, 9999)
             v <- numeric_version(paste0(v, collapse = "."))
           },
           "~=" = v[[1, l]] <- 9999)
  }

  if (v >= "2.6") {
    # model.to_yaml/from_yaml removed in 2.6
    pkgs <- pkgs[names(pkgs) != "pyyaml"]
    return(pkgs)
  }

  if (v >= "2.4") {
    pkgs["Pillow"] <- "Pillow<8.3"
    return(pkgs)
  }

  if (v >= "2.1") {
    pkgs["pyyaml"] <- "pyyaml==3.12"
    pkgs["h5py"] <- "h5py==2.10.0"
    return(pkgs)
  }

  pkgs
}


#  @inheritSection tensorflow::install_tensorflow "Custom Installation" "Apple Silicon" "Additional Packages"
#  @inherit tensorflow::install_tensorflow details
# @inherit tensorflow::install_tensorflow params return references description details sections
# ## everything except 'seealso' to avoid this warning
# ## Warning: Link to unknown topic in inherited text: keras::install_keras
