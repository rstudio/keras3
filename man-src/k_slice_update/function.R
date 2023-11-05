k_slice_update <-
function (inputs, start_indices, updates) 
keras$ops$slice_update(inputs, start_indices, updates)
