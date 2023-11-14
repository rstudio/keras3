Wrapper allowing a stack of RNN cells to behave as a single cell.

@description
Used to implement efficient stacked RNNs.

# Examples

```r
batch_size <- 3
sentence_length <- 5
num_features <- 2
new_shape <- c(batch_size, sentence_length, num_features)
x <- array(1:30, dim = new_shape)

rnn_cells <- lapply(1:2, function(x) layer_lstm_cell(units = 128))
stacked_lstm <- layer_stacked_rnn_cells(rnn_cells)
lstm_layer <- layer_rnn(cell = stacked_lstm)

result <- lstm_layer(x)
```

@param cells
List of RNN cell instances.

@param ...
Passed on to the Python callable

@export
@family cell stacked rnn layers
@family stacked rnn layers
@family rnn layers
@family layers
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/layers/StackedRNNCells>

