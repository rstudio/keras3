
# keras 2.0.6 (unreleased)

- Added [install_keras()] function which installs both TensorFlow and Keras

- Use keras package as default implementation rather than tf.contrib.keras

- Added [serialize_model()] and [unserialize_model()] functions for saving 
  Keras models as 'raw' R objects.

- Automatically convert 64-bit R floats to backend default float type

- Ensure that arrays passed to generator functions are normalized to C-order 


# keras 2.0.5

- Initial CRAN release

