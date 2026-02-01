# Downloads a file from a URL if it not already in the cache.

By default the file at the url `origin` is downloaded to the cache_dir
`~/.keras`, placed in the cache_subdir `datasets`, and given the
filename `fname`. The final location of a file `example.txt` would
therefore be `~/.keras/datasets/example.txt`. Files in `.tar`,
`.tar.gz`, `.tar.bz`, and `.zip` formats can also be extracted.

Passing a hash will verify the file after download. The command line
programs `shasum` and `sha256sum` can compute the hash.

## Usage

``` r
get_file(
  fname = NULL,
  origin = NULL,
  ...,
  file_hash = NULL,
  cache_subdir = "datasets",
  hash_algorithm = "auto",
  extract = FALSE,
  archive_format = "auto",
  cache_dir = NULL,
  force_download = FALSE
)
```

## Arguments

- fname:

  If the target is a single file, this is your desired local name for
  the file. If `NULL`, the name of the file at `origin` will be used. If
  downloading and extracting a directory archive, the provided `fname`
  will be used as extraction directory name (only if it doesn't have an
  extension).

- origin:

  Original URL of the file.

- ...:

  For forward/backward compatability.

- file_hash:

  The expected hash string of the file after download. The sha256 and
  md5 hash algorithms are both supported.

- cache_subdir:

  Subdirectory under the Keras cache dir where the file is saved. If an
  absolute path, e.g. `"/path/to/folder"` is specified, the file will be
  saved at that location.

- hash_algorithm:

  Select the hash algorithm to verify the file. options are `"md5'`,
  `"sha256'`, and `"auto'`. The default 'auto' detects the hash
  algorithm in use.

- extract:

  If `TRUE`, extracts the archive. Only applicable to compressed archive
  files like tar or zip.

- archive_format:

  Archive format to try for extracting the file. Options are `"auto'`,
  `"tar'`, `"zip'`, and `NULL`. `"tar"` includes tar, tar.gz, and tar.bz
  files. The default `"auto"` corresponds to `c("tar", "zip")`. `NULL`
  or an empty list will return no matches found.

- cache_dir:

  Location to store cached files, when `NULL` it defaults to
  `Sys.getenv("KERAS_HOME", "~/.keras/")`.

- force_download:

  If `TRUE`, the file will always be re-downloaded regardless of the
  cache state.

## Value

Path to the downloaded file.

\*\* Warning on malicious downloads \*\*

Downloading something from the Internet carries a risk. NEVER download a
file/archive if you do not trust the source. We recommend that you
specify the `file_hash` argument (if the hash of the source file is
known) to make sure that the file you are getting is the one you expect.

## Examples

    path_to_downloaded_file <- get_file(
        "flower_photos",
        origin = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
        extract = TRUE
    )

## See also

- <https://keras.io/api/utils/python_utils#getfile-function>

Other utils:  
[`audio_dataset_from_directory()`](https://keras3.posit.co/reference/audio_dataset_from_directory.md)  
[`clear_session()`](https://keras3.posit.co/reference/clear_session.md)  
[`config_disable_interactive_logging()`](https://keras3.posit.co/reference/config_disable_interactive_logging.md)  
[`config_disable_traceback_filtering()`](https://keras3.posit.co/reference/config_disable_traceback_filtering.md)  
[`config_enable_interactive_logging()`](https://keras3.posit.co/reference/config_enable_interactive_logging.md)  
[`config_enable_traceback_filtering()`](https://keras3.posit.co/reference/config_enable_traceback_filtering.md)  
[`config_is_interactive_logging_enabled()`](https://keras3.posit.co/reference/config_is_interactive_logging_enabled.md)  
[`config_is_traceback_filtering_enabled()`](https://keras3.posit.co/reference/config_is_traceback_filtering_enabled.md)  
[`get_source_inputs()`](https://keras3.posit.co/reference/get_source_inputs.md)  
[`image_array_save()`](https://keras3.posit.co/reference/image_array_save.md)  
[`image_dataset_from_directory()`](https://keras3.posit.co/reference/image_dataset_from_directory.md)  
[`image_from_array()`](https://keras3.posit.co/reference/image_from_array.md)  
[`image_load()`](https://keras3.posit.co/reference/image_load.md)  
[`image_smart_resize()`](https://keras3.posit.co/reference/image_smart_resize.md)  
[`image_to_array()`](https://keras3.posit.co/reference/image_to_array.md)  
[`layer_feature_space()`](https://keras3.posit.co/reference/layer_feature_space.md)  
[`normalize()`](https://keras3.posit.co/reference/normalize.md)  
[`pad_sequences()`](https://keras3.posit.co/reference/pad_sequences.md)  
[`set_random_seed()`](https://keras3.posit.co/reference/set_random_seed.md)  
[`split_dataset()`](https://keras3.posit.co/reference/split_dataset.md)  
[`text_dataset_from_directory()`](https://keras3.posit.co/reference/text_dataset_from_directory.md)  
[`timeseries_dataset_from_array()`](https://keras3.posit.co/reference/timeseries_dataset_from_array.md)  
[`to_categorical()`](https://keras3.posit.co/reference/to_categorical.md)  
[`zip_lists()`](https://keras3.posit.co/reference/zip_lists.md)  
