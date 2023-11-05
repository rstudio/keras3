constraint_nonneg <-
function () 
{
    args <- capture_args2(NULL)
    do.call(keras$constraints$NonNeg, args)
}
