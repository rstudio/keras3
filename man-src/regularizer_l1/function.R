regularizer_l1 <-
function (l1 = 0.01) 
{
    args <- capture_args2(NULL)
    do.call(keras$regularizers$L1, args)
}
