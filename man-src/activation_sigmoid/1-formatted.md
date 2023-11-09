Sigmoid activation function.

@description
It is defined as: `sigmoid(x) = 1 / (1 + exp(-x))`.

For small values (<-5),
`sigmoid` returns a value close to zero, and for large values (>5)
the result of the function gets close to 1.

Sigmoid is equivalent to a 2-element softmax, where the second element is
assumed to be zero. The sigmoid function always returns a value between
0 and 1.

@param x Input tensor.

@export
@family activation functions
@seealso
+ <https:/keras.io/api/layers/activations#sigmoid-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/activations/sigmoid>
