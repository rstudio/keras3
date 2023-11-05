k_where <-
function (condition, x1 = NULL, x2 = NULL) 
keras$ops$where(condition, x1, x2)
