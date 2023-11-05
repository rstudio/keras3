to_categorical <-
function (x, num_classes = NULL) 
{
    args <- capture_args2(list(x = as_integer_array, num_classes = as_integer))
    do.call(keras$utils$to_categorical, args)
}
