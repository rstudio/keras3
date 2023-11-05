k_while_loop <-
function (cond, body, loop_vars, maximum_iterations = NULL) 
keras$ops$while_loop(cond, body, loop_vars, maximum_iterations)
