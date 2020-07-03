skip_if_not_tensorflow_version <- function(version) {
  if (tensorflow::tf_version() < version)
    skip(paste0("Needs TensorFlow version >= ", version))
}