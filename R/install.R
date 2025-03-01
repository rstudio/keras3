#' Install Keras
#'
#' This function will install Keras along with a selected backend, including all Python dependencies.
#'
#' @param envname Name of or path to a Python virtual environment
#' @param extra_packages Additional Python packages to install alongside Keras
#' @param python_version Passed on to `reticulate::virtualenv_starter()`
#' @param backend Which backend(s) to install. Accepted values include `"tensorflow"`, `"jax"` and `"torch"`
#' @param gpu whether to install a GPU capable version of the backend.
#' @param restart_session Whether to restart the R session after installing (note this will only occur within RStudio).
#' @param ... reserved for future compatibility.
#'
#' @returns No return value, called for side effects.
#'
#' @seealso [`tensorflow::install_tensorflow()`]
#' @export
install_keras <- function(
    envname = "r-keras", ...,
    extra_packages = c("scipy", "pandas", "Pillow", "pydot", "ipython", "tensorflow_datasets"),
    python_version = ">=3.9,<=3.11",
    # backend = "tensorflow",
    backend = c("tensorflow", "jax"),
    # backend = "tf-nightly",
    gpu = NA,
    restart_session = TRUE) {

  if (is.na(gpu)) {

    has_nvidia_gpu <- function() {
      lspci_listed <- tryCatch({
        lspci <- system("lspci", intern = TRUE, ignore.stderr = TRUE)
        any(grepl("nvidia", lspci, ignore.case = TRUE))
      },
      warning = function(w) FALSE,
      error = function(e) FALSE)

      if (lspci_listed)
        return(TRUE)

      # lspci doens't list GPUs on WSL Linux, but nvidia-smi does.
      nvidia_smi_listed <- tryCatch(
        system("nvidia-smi -L", intern = TRUE, ignore.stderr = TRUE),
        warning = function(w) character(),
        error = function(e) character()
      )
      if (isTRUE(any(grepl("^GPU [0-9]: ", nvidia_smi_listed))))
        return(TRUE)
      FALSE
    }

    # don't install tensorflow-metal until it's been updated
    # https://pypi.org/project/tensorflow-metal/#history
    gpu <- (is_linux() && has_nvidia_gpu()) ## ||  is_mac_arm64()
  }

  # keras requires tensorflow be installed still.
  if(!any(grepl("tensorflow|tf-nightly", backend)))
    backend <- c("tensorflow", backend)

  if (isTRUE(gpu)) {
    message("Installing GPU components")
    if (is_mac_arm64()) {
      jax <- c("jax-metal") # ??? do we still need this, "ml-dtypes==0.2.*")
      ## installation of 'tensorflow-metal' is disabled until a new version that
      ## is compatible with TF v2.16 is released.
      # tensorflow <- c("tensorflow", "tensorflow-metal")
    } else if (is_linux()) {
      jax <- c("jax[cuda12]")
      tensorflow <- "tensorflow-cpu"
    }
  } else { # no GPU
    jax <- "jax[cpu]"
    tensorflow <- if(is_linux()) "tensorflow-cpu" else "tensorflow"
  }

  # The "numpy" backend requires that "jax" be installed
  # if("jax" %in% backend && !is.null(extra_packages))
  #   # undeclared dependency, import fails otherwise
  #   append(extra_packages) <- "packaging"

  backend <- unlist(lapply(backend, function(name)
    switch(name,
           jax = jax,
           tensorflow = tensorflow,
           "tf-nightly" = local({
             tensorflow <- sub("tensorflow", "tf-nightly", x = tensorflow, fixed = TRUE)
             replace_val(tensorflow, "tf-nightly-metal", "tensorflow-metal")
           }),
           name)
    ))

  reticulate::virtualenv_create(
    envname = envname,
    version = python_version,
    force = identical(envname, "r-keras"),
    packages = NULL
  )
  extra_packages <- unique(extra_packages)

  if(!any(grepl("^numpy[=><!]?", extra_packages)))
    extra_packages <- c(extra_packages, "numpy<2")

  if (length(extra_packages))
    reticulate::py_install(extra_packages, envname = envname)

  if (length(backend))
    reticulate::py_install(backend, envname = envname)

  reticulate::py_install("keras==3.*", envname = envname)

  if (gpu && is_linux() && !any(startsWith(tensorflow, "tensorflow-cpu"))) {
    configure_nvidia_symlinks(envname = envname)
    configure_ptxas_symlink(envname = envname)
  }

  message("Finished installing Keras!")
  if (restart_session && requireNamespace("rstudioapi", quietly = TRUE) &&
    rstudioapi::hasFun("restartSession")) {
    rstudioapi::restartSession()
  }

  invisible(NULL)
}

is_linux <- function() {
  unname(Sys.info()[["sysname"]] == "Linux")
}

#' Configure a Keras backend
#'
#' @param backend string, can be `"tensorflow"`, `"jax"`, `"numpy"`, or `"torch"`.
#' @param gpu bool, whether to use the GPU.
#'
#' @details
#' These functions allow configuring which backend keras will use.
#' Note that only one backend can be configured at a time.
#'
#' The function should be called after `library(keras3)` and before calling
#' other functions within the package (see below for an example).
#'
#' There is experimental support for changing the backend after keras has initialized.
#' using `config_set_backend()`.
#' ```r
#' library(keras3)
#' use_backend("tensorflow")
#' ```
#' @returns Called primarily for side effects. Returns the provided `backend`, invisibly.
#' @export
use_backend <- function(backend, gpu = NA) {

  if (is_keras_loaded()) {
    if (config_backend() != backend)
      stop("The keras backend must be set before keras has inititialized. Please restart the R session.")
  }

  Sys.setenv(KERAS_BACKEND = backend)

  if (reticulate::py_available()) {
    reticulate::import("os")$environ$update(list(KERAS_BACKEND = backend))
  }


  switch(
    paste0(get_os(), "_", backend),

    macOS_tensorflow = {

      if (is.na(gpu))
        gpu <- FALSE

      if (gpu) {
        # py_require("tensorflow", action = "remove")
        py_require(c("tensorflow", "tensorflow-metal"))
      } else {
        py_require(action = "remove", c("tensorflow-macos", "tensorflow-metal"))
        py_require("tensorflow")
      }

    },

    macOS_jax = {

      py_require(c("tensorflow-metal", "tensorflow-macos"),
                 action = "remove")

      if (is.na(gpu))
        gpu <- TRUE

      if (gpu) {
        py_require(c("tensorflow", "jax", "jax-metal"))
      } else {
        py_require("tensorflow", "jax[cpu]")
      }
    },

    macOS_torch = {
      if(isTRUE(gpu))
        warning("GPU usage not supported on macOS. Please use a different backend to use the GPU (jax)")

      py_require(c("tensorflow-metal", "tensorflow-macos"),
                 action = "remove")

      py_require(c("tensorflow", "torch", "torchvision", "torchaudio"))
    },

    macOS_numpy = {
      py_require(c("tensorflow-metal", "tensorflow-macos"), action = "remove")
      py_require(c("tensorflow", "numpy"))
    },

    Linux_tensorflow = {

      if (is.na(gpu))
        gpu <- has_gpu()

      if (gpu) {
        py_require(action = "remove", c("tensorflow", "tensorflow-cpu"))
        py_require("tensorflow[and-cuda]")
      } else {
        py_require(action = "remove", c("tensorflow", "tensorflow[and-cuda]"))
        py_require("tensorflow-cpu")
      }
    },

    Linux_jax = {
      py_require(c("tensorflow", "tensorflow[and-cuda]"),
                 action = "remove")

      if (is.na(gpu))
        gpu <- has_gpu()

      if (gpu) {
        py_require(c("tensorflow-cpu", "jax[cuda12]"))
      } else {
        py_require(c("tensorflow-cpu", "jax[cpu]"))
      }
    },

    Linux_torch = {
      py_require(c("tensorflow", "tensorflow[and-cuda]"), action = "remove")

      if (is.na(gpu))
        gpu <- has_gpu()

      if (gpu) {
        py_require(c("tensorflow-cpu", "torch", "torchvision", "torchaudio"))
      } else {
        Sys.setenv("UV_INDEX" = "https://download.pytorch.org/whl/cpu")
        py_require(c("tensorflow-cpu", "torch", "torchvision", "torchaudio"))
                   # additional_args = c("--index", "https://download.pytorch.org/whl/cpu"))
      }
    },

    Linux_numpy = {
      py_require(c("tensorflow", "tensorflow[and-cuda]"), action = "remove")
      py_require(c("tensorflow-cpu", "numpy"))
    },

    Windows_tensorflow = {
      if(isTRUE(gpu)) warning("GPU usage not supported on Windows. Please use WSL.")
      py_require(c("tensorflow", "numpy<2"))
    },

    Windows_jax = {
      if(isTRUE(gpu)) warning("GPU usage not supported on Windows. Please use WSL.")
      py_require("jax")
    },

    Windows_torch = {
      if (is.na(gpu))
        gpu <- has_gpu()

      if (gpu) {
        Sys.setenv("UV_INDEX" = "https://download.pytorch.org/whl/cu126")
        py_require(c("torch", "torchvision", "torchaudio"))
                   # additional_args = c("--index", "https://download.pytorch.org/whl/cu126"))
      } else {
        py_require(c("torch", "torchvision", "torchaudio"))
      }
    },

    Windows_numpy = {
      py_require("numpy")
    }
  )

  invisible(backend)
}



get_os <- function() {
  if (is_windows()) "Windows" else if (is_mac_arm64()) "macOS" else "Linux"
}

is_keras_loaded <- function() {
  # package .onLoad() has run (can be FALSE if in devtools::load_all())
  !is.null(keras) &&

  # python is initialized
  reticulate::py_available() &&

  # the keras module proxy has been resolved
  # (reticulate:::py_module_proxy_import()
  #  removes 'module' from the lazy_loaded PyObjectRef module env)
  !exists("module", envir = keras)
}


has_gpu <- function() {

  has_nvidia_gpu <- function() {
    lspci_listed <- tryCatch({
      lspci <- system("lspci", intern = TRUE, ignore.stderr = TRUE)
      any(grepl("nvidia", lspci, ignore.case = TRUE))
    },
    warning = function(w) FALSE,
    error = function(e) FALSE)

    if (lspci_listed)
      return(TRUE)

    # lspci doens't list GPUs on WSL Linux, but nvidia-smi does.
    nvidia_smi_listed <- tryCatch(
      system("nvidia-smi -L", intern = TRUE, ignore.stderr = TRUE),
      warning = function(w) character(),
      error = function(e) character()
    )
    if (isTRUE(any(grepl("^GPU [0-9]: ", nvidia_smi_listed))))
      return(TRUE)
    FALSE
  }

  is_linux() && has_nvidia_gpu()

}


get_py_requirements <- function() {
  python_version <- ">=3.10"
  packages <- "tensorflow"

  if(is_linux()) {

    if(has_gpu()) {
      packages <- "tensorflow[and-cuda]"
    } else {
      packages <- "tensorflow-cpu"
    }

  } else if (is_mac_arm64()) {

    use_gpu <- FALSE
    if (use_gpu) {
      packages <- c("tensorflow-macos", "tensorflow-metal")
      python_version <- ">=3.9,<=3.11"
    }

  } else if (is_windows()) {

  }

  list(packages = packages, python_version = python_version)
}


python_module_dir <- function(python, module, stderr = TRUE) {

  force(python)
  py_cmd <- sprintf("import %s; print(%1$s.__file__)", module)

  module_file <- suppressWarnings(system2(
    python, c("-c", shQuote(py_cmd)),
    stdout = TRUE, stderr = stderr))

  if (!is.null(attr(module_file, "status")) ||
      !is_string(module_file) ||
      !file.exists(module_file))
    return()

  dirname(module_file)

}


configure_nvidia_symlinks <- function(envname) {
  if(!is_linux()) return()
  python <- reticulate::virtualenv_python(envname)

  nvidia_path <- python_module_dir(python, "nvidia")
  if(is.null(nvidia_path)) return()
  # "~/.virtualenvs/r-tensorflow/lib/python3.9/site-packages/nvidia/cudnn"

  nvidia_sos <- Sys.glob(paste0(nvidia_path, "/*/lib/*.so*"))
  if(!length(nvidia_sos)) return()
  # [1] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas.so.12"
  # [2] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cublas/lib/libcublasLt.so.12"
  # [3] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cublas/lib/libnvblas.so.12"
  # [4] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcheckpoint.so"
  # [5] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12"
  # [6] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libnvperf_host.so"
  # [7] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libnvperf_target.so"
  # [8] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libpcsamplingutil.so"
  # [9] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so.12.3"
  # [10] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
  # [11] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
  # [12] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.8"
  # [13] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_adv_infer.so.8"
  # [14] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_adv_train.so.8"
  # [15] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_cnn_infer.so.8"
  # [16] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_cnn_train.so.8"
  # [17] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops_infer.so.8"
  # [18] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn_ops_train.so.8"
  # [19] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cufft/lib/libcufft.so.11"
  # [20] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cufft/lib/libcufftw.so.11"
  # [21] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/curand/lib/libcurand.so.10"
  # [22] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cusolver/lib/libcusolver.so.11"
  # [23] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cusolver/lib/libcusolverMg.so.11"
  # [24] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/cusparse/lib/libcusparse.so.12"
  # [25] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2"
  # [26] "~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12"
  ## we don't need *all* of these, but as of 2.16, in addition to cudnn, we need
  ## libcusparse.so.12 libnvJitLink.so.12 libcusolver.so.11 libcufft.so.11 libcublasLt.so.12 libcublas.so.12
  ## We symlink all of them to (try to be) future proof

  # "~/.virtualenvs/r-tensorflow/lib/python3.9/site-packages/tensorflow"
  tf_lib_path <- python_module_dir(python, "tensorflow", stderr = FALSE)

  from <- sub("^.*/site-packages/", "../", nvidia_sos)
  to <- file.path(tf_lib_path, basename(nvidia_sos))
  writeLines("creating symlinks:")
  writeLines(paste("-", shQuote(to), "->", shQuote(from)))
  # creating symlinks:
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcublas.so.12' -> '../nvidia/cublas/lib/libcublas.so.12'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcublasLt.so.12' -> '../nvidia/cublas/lib/libcublasLt.so.12'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libnvblas.so.12' -> '../nvidia/cublas/lib/libnvblas.so.12'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcheckpoint.so' -> '../nvidia/cuda_cupti/lib/libcheckpoint.so'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcupti.so.12' -> '../nvidia/cuda_cupti/lib/libcupti.so.12'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libnvperf_host.so' -> '../nvidia/cuda_cupti/lib/libnvperf_host.so'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libnvperf_target.so' -> '../nvidia/cuda_cupti/lib/libnvperf_target.so'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libpcsamplingutil.so' -> '../nvidia/cuda_cupti/lib/libpcsamplingutil.so'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libnvrtc-builtins.so.12.3' -> '../nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so.12.3'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libnvrtc.so.12' -> '../nvidia/cuda_nvrtc/lib/libnvrtc.so.12'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcudart.so.12' -> '../nvidia/cuda_runtime/lib/libcudart.so.12'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcudnn.so.8' -> '../nvidia/cudnn/lib/libcudnn.so.8'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcudnn_adv_infer.so.8' -> '../nvidia/cudnn/lib/libcudnn_adv_infer.so.8'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcudnn_adv_train.so.8' -> '../nvidia/cudnn/lib/libcudnn_adv_train.so.8'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcudnn_cnn_infer.so.8' -> '../nvidia/cudnn/lib/libcudnn_cnn_infer.so.8'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcudnn_cnn_train.so.8' -> '../nvidia/cudnn/lib/libcudnn_cnn_train.so.8'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcudnn_ops_infer.so.8' -> '../nvidia/cudnn/lib/libcudnn_ops_infer.so.8'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcudnn_ops_train.so.8' -> '../nvidia/cudnn/lib/libcudnn_ops_train.so.8'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcufft.so.11' -> '../nvidia/cufft/lib/libcufft.so.11'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcufftw.so.11' -> '../nvidia/cufft/lib/libcufftw.so.11'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcurand.so.10' -> '../nvidia/curand/lib/libcurand.so.10'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcusolver.so.11' -> '../nvidia/cusolver/lib/libcusolver.so.11'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcusolverMg.so.11' -> '../nvidia/cusolver/lib/libcusolverMg.so.11'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libcusparse.so.12' -> '../nvidia/cusparse/lib/libcusparse.so.12'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libnccl.so.2' -> '../nvidia/nccl/lib/libnccl.so.2'
  # - '~/.virtualenvs/r-tensorflow/lib/python3.10/site-packages/tensorflow/libnvJitLink.so.12' -> '../nvidia/nvjitlink/lib/libnvJitLink.so.12'
  # - '~/.virtualenvs/r-tensorflow/bin/ptxas' -> '../../lib/python3.10/site-packages/nvidia/cuda_nvcc/bin/ptxas'
  file.symlink(from = from, to = to)

}

configure_ptxas_symlink <- function(envname = "r-keras") {
  if(!is_linux()) return()
  python <- reticulate::virtualenv_python(envname)

  nvcc_path <- python_module_dir(python, "nvidia.cuda_nvcc")
  if(is.null(nvcc_path)) return()

  # configure a link so that ptxas can be found on the PATH
  # when the venv is activated.
  # https://discuss.tensorflow.org/t/tensorflow-version-2-16-just-released/23140/6#resolving-the-ptxas-issue-3
  nvcc_bins <- Sys.glob(file.path(nvcc_path, "bin/*"))
  if(!length(nvcc_bins)) return()
  # "~/.virtualenvs/r-tensorflow/lib/python3.9/site-packages/nvidia/cuda_nvcc/bin/ptxas"

  to <- file.path(dirname(python), basename(nvcc_bins))
  # "~/.virtualenvs/r-tensorflow/bin/ptxas"

  # fs::path_rel(nvcc_bins, to)
  from <- sub(dirname(dirname(python)), "../..", nvcc_bins)
  # "../../lib/python3.9/site-packages/nvidia/cuda_nvcc/bin/ptxas"

  # writeLines("creating symlinks:")
  writeLines(paste("-", shQuote(to), "->", shQuote(from)))
  # '~/.virtualenvs/r-tensorflow/bin/ptxas' -> '../../lib/python3.9/site-packages/nvidia/cuda_nvcc/bin/ptxas'

  file.symlink(from = from, to = to)

}
