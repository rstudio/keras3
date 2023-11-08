#' Install Keras
#'
#' This function will install Keras, including all Python dependencies.
#'
#' @param envname name of or path to a python virtual environment
#' @param extra_packages additional python packages to install alongside keras
#' @param python_version passed on to `reticulate::virtualenv_starter()`
#' @param ... reserved for future compatability.
#'
#' @seealso [`tensorflow::install_tensorflow()`]
#' @export
install_keras <- function(...,
                          envname = "r-keras",
                          extra_packages = NULL,
                          python_version = "3.10",
                          devel = FALSE
                          # backend = c("tensorflow", "jax", "pytorch")
                          ) {

  python <- envname |>
    reticulate::virtualenv_create(version = python_version,
                                  force = identical(envname, "r-keras"),
                                  packages = NULL) |>
    reticulate::virtualenv_python()

  withr::local_path(dirname(python))
  system2 <- reticulate:::system2t


  # message("DONE!")
  # return()
  # system2("python", "-m pip install tensorflow")

  if(Sys.getenv("CI") == "true") {
    system2("python", "-m pip install tf-nightly keras-nightly")
    return()
    # system2("python", "-m pip uninstall keras")
    # system2("python", "-m pip install keras-nightly tf-nightly")
  }

  if(!devel) {
    system2("python", "-m pip install keras-nightly tf-nightly")
    message("DONE")
    return()
  }


  # withr::local_dir(withr::local_tempdir())
  keras_team_keras_dir <- "~/github/keras-team/keras"
  # if(TRUE) {
  if(!dir.exists(keras_team_keras_dir)) {
    keras_team_keras_dir <- tempfile(pattern = "keras-team-keras-")
    dir.create(keras_team_keras_dir)
    system2("git", c(
      "clone --depth 1 --branch master https://github.com/keras-team/keras",
      keras_team_keras_dir))
    withr::defer(unlink(keras_team_keras_dir, recursive = TRUE))
  }
  withr::local_dir(keras_team_keras_dir)
  unlink("tmp_build_dir", recursive = TRUE)
  # browser()
  system2("git", "pull")
  # system2("python", c("-m pip install -r requirements.txt")) # unpin tf-nightly for Python 3.12
  system2("python", c("-m pip install -r requirements-common.txt"))
  system2("python", c("-m pip install ipython")) # for interactive debugging
  system2("python", c("-m pip install openai tiktoken")) # for roxygen generation
  system2("python", c("-m pip install tf-nightly jax[cpu]")) # unpin tf-nightly for Python 3.12
  system2("python", c("-m pip uninstall -y keras keras-nightly"))
  system2("python", c("-m pip install torch torchvision")) # needed for pip_build.py?? (but why?)

  system2("python", c("pip_build.py --install"))
  system2("python", c("-m pip uninstall -y torch torchvision"))
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
