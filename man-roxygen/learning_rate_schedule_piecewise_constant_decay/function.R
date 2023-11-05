learning_rate_schedule_piecewise_constant_decay <-
function (boundaries, values, name = "PiecewiseConstant") 
{
    args <- capture_args2(NULL)
    do.call(keras$optimizers$schedules$PiecewiseConstantDecay, 
        args)
}
