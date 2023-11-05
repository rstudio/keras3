k_clip <-
function (x, x_min, x_max) 
keras$ops$clip(x, x_min, x_max)
