k_binary_crossentropy <-
function (target, output, from_logits = FALSE) 
keras$ops$binary_crossentropy(target, output, from_logits)
