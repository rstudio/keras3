learning_rate_schedule_exponential_decay <-
function (initial_learning_rate, decay_steps, decay_rate, staircase = FALSE, 
    name = "ExponentialDecay") 
{
    args <- capture_args2(list(decay_steps = as_integer))
    do.call(keras$optimizers$schedules$ExponentialDecay, args)
}
