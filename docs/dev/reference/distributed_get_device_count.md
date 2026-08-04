# Count available devices of one type

When `device_type` is not provided, Keras counts devices of the default
available type. Device types are not mixed in a single count. Keras
3.15.1 provides the backend implementation for JAX.

## Usage

``` r
distributed_get_device_count(device_type = NULL)
```

## Arguments

- device_type:

  String, one of `"cpu"`, `"gpu"`, or `"tpu"`. If `NULL`, Keras counts
  GPU or TPU devices when available and otherwise counts CPU devices.

## Value

An integer number of devices.

## Examples

    distributed_get_device_count("cpu")
