regularizer_l1_l2 <-
function (l1 = 0, l2 = 0) 
{
    args <- capture_args2(NULL)
    do.call(keras$regularizers$L1L2, args)
}
