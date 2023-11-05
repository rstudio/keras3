config_enable_unsafe_deserialization <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$config$enable_unsafe_deserialization, args)
}
