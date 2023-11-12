One-stop utility for preprocessing and encoding structured data.

@description
**Available feature types:**

Note that all features can be referred to by their string name,
e.g. `"integer_categorical"`. When using the string name, the default
argument values are used.


```r
# Plain float values.
feature_float(name = NULL)

# Float values to be preprocessed via featurewise standardization
# (i.e. via a `layer_normalization()` layer).
feature_float_normalized(name = NULL)

# Float values to be preprocessed via linear rescaling
# (i.e. via a `layer_rescaling` layer).
feature_float_rescaled(scale = 1., offset = 0., name = NULL)

# Float values to be discretized. By default, the discrete
# representation will then be one-hot encoded.
feature_float_discretized(
  num_bins,
  bin_boundaries = NULL,
  output_mode = "one_hot",
  name = NULL
)

# Integer values to be indexed. By default, the discrete
# representation will then be one-hot encoded.
feature_integer_categorical(
  max_tokens = NULL,
  num_oov_indices = 1,
  output_mode = "one_hot",
  name = NULL
)

# String values to be indexed. By default, the discrete
# representation will then be one-hot encoded.
feature_string_categorical(
  max_tokens = NULL,
  num_oov_indices = 1,
  output_mode = "one_hot",
  name = NULL
)

# Integer values to be hashed into a fixed number of bins.
# By default, the discrete representation will then be one-hot encoded.
feature_integer_hashed(num_bins, output_mode = "one_hot", name = NULL)

# String values to be hashed into a fixed number of bins.
# By default, the discrete representation will then be one-hot encoded.
feature_string_hashed(num_bins, output_mode = "one_hot", name = NULL)
```
# Examples
**Basic usage with a dict of input data:**


```r
raw_data <- list(
  float_values = c(0.0, 0.1, 0.2, 0.3),
  string_values = c("zero", "one", "two", "three"),
  int_values = as.integer(c(0, 1, 2, 3))
)

dataset <- tfdatasets::tensor_slices_dataset(raw_data)

feature_space <- layer_feature_space(
  features = list(
    float_values = "float_normalized",
    string_values = "string_categorical",
    int_values = "integer_categorical"
  ),
  crosses = list(c("string_values", "int_values")),
  output_mode = "concat"
)

# Before you start using the feature_space(),
# you must `adapt()` it on some data.
feature_space |> adapt(dataset)

# You can call the feature_space() on a named list of
# data (batched or unbatched).
output_vector <- feature_space(raw_data)
```

**Basic usage with `tf.data`:**


```r
library(tfdatasets)
# Unlabeled data
preprocessed_ds <- unlabeled_dataset |>
  dataset_map(feature_space)

# Labeled data
preprocessed_ds <- labeled_dataset |>
  dataset_map(function(x, y) tuple(feature_space(x), y))
```

**Basic usage with the Keras Functional API:**


```r
# Retrieve a named list of Keras layer_input() objects
(inputs <- feature_space$get_inputs())
```

```
## $float_values
## <KerasTensor shape=(None, 1), dtype=float32, sparse=None, name=float_values>
##
## $string_values
## <KerasTensor shape=(None, 1), dtype=string, sparse=None, name=string_values>
##
## $int_values
## <KerasTensor shape=(None, 1), dtype=int32, sparse=None, name=int_values>
```

```r
# Retrieve the corresponding encoded Keras tensors
(encoded_features <- feature_space$get_encoded_features())
```

```
## <KerasTensor shape=(None, 43), dtype=float32, sparse=False, name=keras_tensor_7>
```

```r
# Build a Functional model
outputs <- encoded_features |> layer_dense(1, activation = "sigmoid")
model <- keras_model(inputs, outputs)
```

**Customizing each feature or feature cross:**


```r
feature_space <- layer_feature_space(
  features = list(
    float_values = feature_float_normalized(),
    string_values = feature_string_categorical(max_tokens = 10),
    int_values = feature_integer_categorical(max_tokens = 10)
  ),
  crosses = list(
    feature_cross(c("string_values", "int_values"), crossing_dim = 32)
  ),
  output_mode = "concat"
)
```

**Returning a dict of integer-encoded features:**


```r
feature_space <- layer_feature_space(
  features = list(
    "string_values" = feature_string_categorical(output_mode = "int"),
    "int_values" = feature_integer_categorical(output_mode = "int")
  ),
  crosses = list(
    feature_cross(
      feature_names = c("string_values", "int_values"),
      crossing_dim = 32,
      output_mode = "int"
    )
  ),
  output_mode = "dict"
)
```

**Specifying your own Keras preprocessing layer:**


```r
# Let's say that one of the features is a short text paragraph that
# we want to encode as a vector (one vector per paragraph) via TF-IDF.
data <- list(text = c("1st string", "2nd string", "3rd string"))

# There's a Keras layer for this: layer_text_vectorization()
custom_layer <- layer_text_vectorization(output_mode = "tf_idf")

# We can use feature_custom() to create a custom feature
# that will use our preprocessing layer.
feature_space <- layer_feature_space(
  features = list(
    text = feature_custom(preprocessor = custom_layer,
                          dtype = "string",
                          output_mode = "float"
    )
  ),
  output_mode = "concat"
)
feature_space |> adapt(tfdatasets::tensor_slices_dataset(data))
output_vector <- feature_space(data)
```

**Retrieving the underlying Keras preprocessing layers:**


```r
# The preprocessing layer of each feature is available in `$preprocessors`.
preprocessing_layer <- feature_space$preprocessors$feature1

# The crossing layer of each feature cross is available in `$crossers`.
# It's an instance of layer_hashed_crossing()
crossing_layer <- feature_space$crossers[["feature1_X_feature2"]]
```

**Saving and reloading a FeatureSpace:**


```r
feature_space$save("featurespace.keras")
reloaded_feature_space <- keras$models$load_model("featurespace.keras")
```

@param feature_names Dict mapping the names of your features to their
    type specification, e.g. `{"my_feature": "integer_categorical"}`
    or `{"my_feature": FeatureSpace.integer_categorical()}`.
    For a complete list of all supported types, see
    "Available feature types" paragraph below.
@param output_mode One of `"concat"` or `"dict"`. In concat mode, all
    features get concatenated together into a single vector.
    In dict mode, the FeatureSpace returns a dict of individually
    encoded features (with the same keys as the input dict keys).
@param crosses List of features to be crossed together, e.g.
    `crosses=[("feature_1", "feature_2")]`. The features will be
    "crossed" by hashing their combined value into
    a fixed-length vector.
@param crossing_dim Default vector size for hashing crossed features.
    Defaults to `32`.
@param hashing_dim Default vector size for hashing features of type
    `"integer_hashed"` and `"string_hashed"`. Defaults to `32`.
@param num_discretization_bins Default number of bins to be used for
    discretizing features of type `"float_discretized"`.
    Defaults to `32`.
@param name String, name for the object
@param object see description
@param features see description

@export
@family preprocessing layers
@seealso
+ <https:/keras.io/api/utils/feature_space#featurespace-class>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/FeatureSpace>
