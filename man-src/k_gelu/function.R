k_gelu <-
function (x, approximate = TRUE) 
keras$ops$gelu(x, approximate)
