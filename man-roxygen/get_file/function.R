get_file <-
function (fname = NULL, origin = NULL, untar = FALSE, md5_hash = NULL, 
    file_hash = NULL, cache_subdir = "datasets", hash_algorithm = "auto", 
    extract = FALSE, archive_format = "auto", cache_dir = NULL) 
{
    args <- capture_args2(NULL)
    do.call(keras$utils$get_file, args)
}
