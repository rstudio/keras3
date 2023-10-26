#' Torch module wrapper layer.
#'
#' @description
#' `TorchModuleWrapper` is a wrapper class that can turn any
#' `torch.nn.Module` into a Keras layer, in particular by making its
#' parameters trackable by Keras.
#'
#' # Examples
#' Here's an example of how the `TorchModuleWrapper` can be used with vanilla
#' PyTorch modules.
#'
#' ```python
#' import torch.nn as nn
#' import torch.nn.functional as F
#'
#' import keras
#' from keras.layers import TorchModuleWrapper
#'
#' class Classifier(keras.Model):
#'     def __init__(self, **kwargs):
#'         super().__init__(**kwargs)
#'         # Wrap `torch.nn.Module`s with `TorchModuleWrapper`
#'         # if they contain parameters
#'         self.conv1 = TorchModuleWrapper(
#'             nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))
#'         )
#'         self.conv2 = TorchModuleWrapper(
#'             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
#'         )
#'         self.pool = nn.MaxPool2d(kernel_size=(2, 2))
#'         self.flatten = nn.Flatten()
#'         self.dropout = nn.Dropout(p=0.5)
#'         self.fc = TorchModuleWrapper(nn.Linear(1600, 10))
#'
#'     def call(self, inputs):
#'         x = F.relu(self.conv1(inputs))
#'         x = self.pool(x)
#'         x = F.relu(self.conv2(x))
#'         x = self.pool(x)
#'         x = self.flatten(x)
#'         x = self.dropout(x)
#'         x = self.fc(x)
#'         return F.softmax(x, dim=1)
#'
#' model = Classifier()
#' model.build((1, 28, 28))
#' print("Output shape:", model(torch.ones(1, 1, 28, 28).to("cuda")).shape)
#'
#' model.compile(
#'     loss="sparse_categorical_crossentropy",
#'     optimizer="adam",
#'     metrics=["accuracy"]
#' )
#' model.fit(train_loader, epochs=5)
#' ```
#'
#' @param module `torch.nn.Module` instance. If it's a `LazyModule`
#'     instance, then its parameters must be initialized before
#'     passing the instance to `TorchModuleWrapper` (e.g. by calling
#'     it once).
#' @param name The name of the layer (string).
#' @param object Object to compose the layer with. A tensor, array, or sequential model.
#' @param ... Passed on to the Python callable
#'
#' @export
#' @seealso
#' + <https://www.tensorflow.org/api_docs/python/tf/keras/layers/TorchModuleWrapper>
layer_torch_module_wrapper <-
function (object, module, name = NULL, ...)
{
    args <- capture_args2(list(input_shape = normalize_shape,
        batch_size = as_integer, batch_input_shape = normalize_shape),
        ignore = "object")
    create_layer(keras$layers$TorchModuleWrapper, object, args)
}
