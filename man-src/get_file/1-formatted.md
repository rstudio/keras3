Downloads a file from a URL if it not already in the cache.

@description
By default the file at the url `origin` is downloaded to the
cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
and given the filename `fname`. The final location of a file
`example.txt` would therefore be `~/.keras/datasets/example.txt`.
Files in `.tar`, `.tar.gz`, `.tar.bz`, and `.zip` formats can
also be extracted.

Passing a hash will verify the file after download. The command line
programs `shasum` and `sha256sum` can compute the hash.

# Examples
```python
path_to_downloaded_file = get_file(
    origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    extract=True,
)
```

@returns
Path to the downloaded file.

** Warning on malicious downloads **

Downloading something from the Internet carries a risk.
NEVER download a file/archive if you do not trust the source.
We recommend that you specify the `file_hash` argument
(if the hash of the source file is known) to make sure that the file you
are getting is the one you expect.

@param fname Name of the file. If an absolute path, e.g. `"/path/to/file.txt"`
    is specified, the file will be saved at that location.
    If `None`, the name of the file at `origin` will be used.
@param origin Original URL of the file.
@param untar Deprecated in favor of `extract` argument.
    boolean, whether the file should be decompressed
@param md5_hash Deprecated in favor of `file_hash` argument.
    md5 hash of the file for verification
@param file_hash The expected hash string of the file after download.
    The sha256 and md5 hash algorithms are both supported.
@param cache_subdir Subdirectory under the Keras cache dir where the file is
    saved. If an absolute path, e.g. `"/path/to/folder"` is
    specified, the file will be saved at that location.
@param hash_algorithm Select the hash algorithm to verify the file.
    options are `"md5'`, `"sha256'`, and `"auto'`.
    The default 'auto' detects the hash algorithm in use.
@param extract True tries extracting the file as an Archive, like tar or zip.
@param archive_format Archive format to try for extracting the file.
    Options are `"auto'`, `"tar'`, `"zip'`, and `None`.
    `"tar"` includes tar, tar.gz, and tar.bz files.
    The default `"auto"` corresponds to `["tar", "zip"]`.
    None or an empty list will return no matches found.
@param cache_dir Location to store cached files, when None it
    defaults ether `$KERAS_HOME` if the `KERAS_HOME` environment
    variable is set or `~/.keras/`.

@export
@family utils
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file>
