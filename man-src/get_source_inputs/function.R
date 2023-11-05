get_source_inputs <-
function (tensor) 
{
    args <- capture_args2(NULL)
    do.call(keras$utils$get_source_inputs, args)
}
