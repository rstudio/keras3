Wrapper allowing a stack of RNN cells to behave as a single cell.

Used to implement efficient stacked RNNs.

Args:
  cells: List of RNN cell instances.

Examples:

```python
batch_size = 3
sentence_length = 5
num_features = 2
new_shape = (batch_size, sentence_length, num_features)
x = np.reshape(np.arange(30), new_shape)

rnn_cells = [keras.layers.LSTMCell(128) for _ in range(2)]
stacked_lstm = keras.layers.StackedRNNCells(rnn_cells)
lstm_layer = keras.layers.RNN(stacked_lstm)

result = lstm_layer(x)
```
