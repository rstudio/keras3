#! /usr/bin/env Rscript



for (nm in c("tf-nightly", "tf-nightly-cpu"))
  keras::install_keras(
    version = nm,
    envname = nm,
    method = "conda",
    restart_session = FALSE
  )
