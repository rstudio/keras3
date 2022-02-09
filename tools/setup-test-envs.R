#! /usr/bin/env Rscript

args <- commandArgs(TRUE)

if("--fresh" %in% args) # fresh start
  unlink(reticulate::miniconda_path(), recursive = TRUE)

remotes::install_github("rstudio/reticulate", force = TRUE)

ver <- as.data.frame(rbind(
  c(py = "3.8", tf = "2.8"),
  c(py = "3.9", tf = "2.7"),
  c(py = "3.8", tf = "2.6"),
  c(py = "3.7", tf = "2.5"),
  c(py = "3.7", tf = "2.4"),
  # c(py = "3.6", tf = "2.3"),
  # c(py = "3.6", tf = "2.2"),
  # c(py = "3.6", tf = "2.1"),
  # c(py = "3.6", tf =   "1"),
  # c(py = "3.8", tf = "2.8.0rc0"),
  c(py = "3.9", tf = "nightly")))

ver$tf <- paste0(ver$tf, "-cpu")
ver$name <- paste0("tf-", sub(".0rc[0-9]+", "", ver$tf))

if(!reticulate:::miniconda_exists())
  reticulate::install_miniconda()


for (i in seq_len(nrow(ver))) {
  # tensorflow::install_tensorflow(
  keras::install_keras(
    version = ver$tf[i],
    envname = ver$name[i],
    conda_python_version = ver$py[i],
    pip_ignore_installed = TRUE,
    extra_packages = "ipython",
    method = "conda",
    restart_session = FALSE
  ) # |> try()
}


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
