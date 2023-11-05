learning_rate_schedule_polynomial_decay <-
function (initial_learning_rate, decay_steps, end_learning_rate = 1e-04, 
    power = 1, cycle = FALSE, name = "PolynomialDecay") 
{
    args <- capture_args2(list(decay_steps = as_integer))
    do.call(keras$optimizers$schedules$PolynomialDecay, args)
}
