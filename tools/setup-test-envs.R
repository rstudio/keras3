#! /usr/bin/env Rscript

args <- commandArgs(TRUE)

if("--fresh" %in% args) # fresh start
  unlink(reticulate::miniconda_path(), recursive = TRUE)

vers <- c("2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "nightly")
vers <- paste0(vers, "-cpu")

names(vers) <- paste0("tf-", vers)
names(vers) <- sub(".0rc[0-9]+", "", names(vers))


if(!reticulate:::miniconda_exists())
  reticulate::install_miniconda()

for (i in seq_along(vers))
  keras::install_keras(
    version = vers[i],
    envname = names(vers)[i],
    method = "conda", restart_session = FALSE)
