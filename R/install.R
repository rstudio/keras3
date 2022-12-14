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
install_keras <- function(method = c("auto", "virtualenv", "conda"),
                          conda = "auto",
                          version = "default",
                          tensorflow = version,
                          extra_packages = NULL,
                          ...,
                          pip_ignore_installed = TRUE) {
  method <- match.arg(method)

  if(is_mac_arm64()) {
    return(tensorflow::install_tensorflow(
      method = method,
      conda = conda,
      version = version,
      extra_packages = c(default_extra_packages(),
                         extra_packages),
      ...))
  }

  pkgs <- default_extra_packages(tensorflow)
  if(!is.null(extra_packages)) # user supplied package version constraints take precedence
    pkgs[gsub("[=<>~]{1,2}[0-9.]+$", "", extra_packages)] <- extra_packages

  if(tensorflow %in% c("cpu", "gpu"))
    tensorflow <- paste0("default-", tensorflow)

  if(grepl("^default", tensorflow))
    tensorflow <- sub("^default", as.character(default_version), tensorflow)

  tensorflow::install_tensorflow(
    method = method,
    conda = conda,
    version = tensorflow,
    extra_packages = pkgs,
    pip_ignore_installed = pip_ignore_installed,
    ...
  )
}

default_version <- numeric_version("2.11")

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
