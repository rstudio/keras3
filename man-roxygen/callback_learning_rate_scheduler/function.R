callback_learning_rate_scheduler <-
function (schedule, verbose = 0L) 
{
    args <- capture_args2(list(schedule = as_integer, verbose = as_integer))
    do.call(keras$callbacks$LearningRateScheduler, args)
}
