#! /usr/bin/env Rscript

args <- commandArgs(TRUE)

if("--fresh" %in% args) # fresh start
  unlink(reticulate::miniconda_path(), recursive = TRUE)

tf_vers <- c("2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "nightly")
tf_vers <- paste0(tf_vers, "-cpu")
py_vers <- c("3.6", "3.6", "3.7", "3.8", "3.8", "3.9", "3.9", "3.9")

names(tf_vers) <- paste0("tf-", tf_vers)
names(tf_vers) <- sub(".0rc[0-9]+", "", names(tf_vers))


if(!reticulate:::miniconda_exists())
  reticulate::install_miniconda()

for (i in seq_along(tf_vers))
    keras::install_keras(
  # tensorflow::install_tensorflow(
    version = tf_vers[i],
    envname = names(tf_vers)[i],
    conda_python_version = py_vers[i],
    pip_ignore_installed = TRUE,
    method = "conda", restart_session = FALSE) # |> try()


# work around conda run bug: https://github.com/conda/conda/issues/10972
# reticulate:::pip_install(
#   "~/.local/share/r-miniconda/envs/tf-2.4-cpu/bin/python",
#   packages = c("tensorflow==2.4.*", keras:::default_extra_packages("2.4")),
#   conda = FALSE, ignore_installed = TRUE)
#
# reticulate:::pip_install(
#   "~/.local/share/r-miniconda/envs/tf-2.5-cpu/bin/python",
#   packages = c("tensorflow==2.5.*", keras:::default_extra_packages("2.5")),
#   conda = FALSE, ignore_installed = TRUE)


# reticulate:::conda_run("python", c("-m", "pip", "install", "--upgrade", "tensorflow-cpu==2.4.*"),
#                        envname = "tf-2.4-cpu")


# ~/.local/share/r-miniconda/bin/conda run --name tf-2.5-cpu python -m pip install --upgrade 'tensorflow-cpu==2.5.*' tensorflow-hub scipy requests pyyaml 'Pillow<8.3' 'h5py pandas
