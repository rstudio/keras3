# Applies a series of layers to an input.

This class is useful to build a preprocessing pipeline, in particular an
image data augmentation pipeline. Compared to a `Sequential` model,
`Pipeline` features a few important differences:

- It's not a `Model`, just a plain layer.

- When the layers in the pipeline are compatible with `tf.data`, the
  pipeline will also remain `tf.data` compatible. That is to say, the
  pipeline will not attempt to convert its inputs to backend-native
  tensors when in a tf.data context (unlike a `Sequential` model).

## Usage

``` r
layer_pipeline(layers, name = NULL)
```

## Arguments

- layers:

  A list of layers.

- name:

  String, name for the object

## Examples

    preprocessing_pipeline <- layer_pipeline(c(
      layer_auto_contrast(, ),
      layer_random_zoom(, 0.2),
      layer_random_rotation(, 0.2)
    ))

    # `ds` is a tf.data.Dataset of images
    ds <- tfdatasets::tensor_slices_dataset(1:100) |>
      tfdatasets::dataset_map(\(.x) {
        random_normal(c(28, 28))
      }) |>
      tfdatasets::dataset_batch(32)
      #|>
      # tfdatasets::dataset_take(4) |>
      # iterate() |> str()

    preprocessed_ds <- ds |>
      tfdatasets::dataset_map(preprocessing_pipeline, num_parallel_calls = 4)
