regularizer_l2 <-
function (l2 = 0.01) 
{
    args <- capture_args2(NULL)
    do.call(keras$regularizers$L2, args)
}
