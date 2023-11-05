k_scatter_update <-
function (inputs, indices, updates) 
keras$ops$scatter_update(inputs, indices, updates)
