initializer_identity <-
function (gain = 1) 
{
    args <- capture_args2(NULL)
    do.call(keras$initializers$IdentityInitializer, args)
}
