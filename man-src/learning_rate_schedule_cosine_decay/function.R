learning_rate_schedule_cosine_decay <-
function (initial_learning_rate, decay_steps, alpha = 0, name = "CosineDecay", 
    warmup_target = NULL, warmup_steps = 0L) 
{
    args <- capture_args2(list(decay_steps = as_integer, warmup_steps = as_integer))
    do.call(keras$optimizers$schedules$CosineDecay, args)
}
