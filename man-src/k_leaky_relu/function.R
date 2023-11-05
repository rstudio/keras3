k_leaky_relu <-
function (x, negative_slope = 0.2) 
keras$ops$leaky_relu(x, negative_slope)
