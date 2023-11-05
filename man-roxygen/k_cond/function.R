k_cond <-
function (pred, true_fn, false_fn) 
keras$ops$cond(pred, true_fn, false_fn)
