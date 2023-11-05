layer_feature_space <-
function (object, features, output_mode = "concat", crosses = NULL, 
    crossing_dim = 32L, hashing_dim = 32L, num_discretization_bins = 32L, 
    name = NULL, feature_names = NULL) 
{
    args <- capture_args2(list(crossing_dim = as_integer, hashing_dim = as_integer, 
        num_discretization_bins = as_integer, feature_names = as_integer), 
        ignore = "object")
    create_layer(keras$utils$FeatureSpace, object, args)
}
