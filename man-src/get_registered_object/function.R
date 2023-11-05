get_registered_object <-
function (name, custom_objects = NULL, module_objects = NULL) 
{
    args <- capture_args2(NULL)
    do.call(keras$utils$get_registered_object, args)
}
