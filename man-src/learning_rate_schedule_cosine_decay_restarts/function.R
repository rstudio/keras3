learning_rate_schedule_cosine_decay_restarts <-
function (initial_learning_rate, first_decay_steps, t_mul = 2, 
    m_mul = 1, alpha = 0, name = "SGDRDecay") 
{
    args <- capture_args2(list(first_decay_steps = as_integer))
    do.call(keras$optimizers$schedules$CosineDecayRestarts, args)
}
