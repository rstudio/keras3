clean_python_tmp_dir <- function() {

  if(!reticulate::py_available())
    # python never initialized, nothing to do
    return()

  python_temp_dir <- dirname(reticulate::py_run_string(
    "import tempfile; x=tempfile.NamedTemporaryFile().name",
    local = TRUE
  )$x)

  detritus <- list.files(
    path = python_temp_dir,
    pattern = "__autograph_generated_file|__pycache__",
    full.names = TRUE,
    all.files = TRUE
  )

  unlink(detritus, recursive = TRUE)
}

withr::defer(clean_python_tmp_dir(), teardown_env())
