random_seed_generator <-
function (seed = NULL, ...) 
{
    args <- capture_args2(list(seed = as_integer))
    do.call(keras$random$SeedGenerator, args)
}
