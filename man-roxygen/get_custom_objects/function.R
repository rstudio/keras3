get_custom_objects <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$utils$get_custom_objects, args)
}
