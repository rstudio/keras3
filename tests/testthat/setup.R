
trace_tempfile_mkdtemp <- function() {

  py_created_tempdirs <- NULL

  on_py_init <- function() {

    py_created_tempdirs <<-
      reticulate::py_run_string(
        local = TRUE, convert = FALSE, "
def patch_mkdtemp():
    from functools import wraps
    from os.path import abspath
    import tempfile

    created_tempdirs = []
    og_mkdtemp = tempfile.mkdtemp

    @wraps(og_mkdtemp)
    def mkdtemp(*args, **kwargs):
        out = og_mkdtemp(*args, **kwargs)
        created_tempdirs.append(abspath(out))
        return out

    tempfile.mkdtemp = mkdtemp
    return created_tempdirs

")$patch_mkdtemp()

  }


  if(reticulate::py_available()) on_py_init()
  else setHook("reticulate.onPyInit", on_py_init)

  function() py_created_tempdirs
}

get_py_created_tempdirs <- trace_tempfile_mkdtemp()

clean_python_tempdirs <- function() {

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

  unlink(unlist(py_to_r(get_py_created_tempdirs())),
         recursive = TRUE)

}

withr::defer(clean_python_tempdirs(), teardown_env())
