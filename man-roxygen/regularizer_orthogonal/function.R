regularizer_orthogonal <-
function (factor = 0.01, mode = "rows") 
{
    args <- capture_args2(NULL)
    do.call(keras$regularizers$OrthogonalRegularizer, args)
}
