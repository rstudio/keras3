learning_rate_schedule_inverse_time_decay <-
function (initial_learning_rate, decay_steps, decay_rate, staircase = FALSE, 
    name = "InverseTimeDecay") 
{
    args <- capture_args2(NULL)
    do.call(keras$optimizers$schedules$InverseTimeDecay, args)
}
