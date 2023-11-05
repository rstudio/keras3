k_categorical_crossentropy <-
function (target, output, from_logits = FALSE, axis = -1L) 
{
    args <- capture_args2(list(axis = as_axis))
    do.call(keras$ops$categorical_crossentropy, args)
}
