Creates a dataset of sliding windows over a timeseries provided as array.

@description
This function takes in a sequence of data-points gathered at
equal intervals, along with time series parameters such as
length of the sequences/windows, spacing between two sequence/windows, etc.,
to produce batches of timeseries inputs and targets.

@returns
A `tf$data$Dataset` instance. If `targets` was passed, the dataset yields
list `(batch_of_sequences, batch_of_targets)`. If not, the dataset yields
only `batch_of_sequences`.

Example 1:

Consider indices `[0, 1, ... 98]`.
With `sequence_length=10,  sampling_rate=2, sequence_stride=3`,
`shuffle=FALSE`, the dataset will yield batches of sequences
composed of the following indices:

```
First sequence:  [0  2  4  6  8 10 12 14 16 18]
Second sequence: [3  5  7  9 11 13 15 17 19 21]
Third sequence:  [6  8 10 12 14 16 18 20 22 24]
...
Last sequence:   [78 80 82 84 86 88 90 92 94 96]
```

In this case the last 2 data points are discarded since no full sequence
can be generated to include them (the next sequence would have started
at index 81, and thus its last step would have gone over 98).

Example 2: Temporal regression.

Consider an array `data` of scalar values, of shape `(steps,)`.
To generate a dataset that uses the past 10
timesteps to predict the next timestep, you would use:


```r
data <- k_array(1:20)
input_data <- data[1:10]
targets <- data[11:20]
dataset <- timeseries_dataset_from_array(
  input_data, targets, sequence_length=10)
iter <- reticulate::as_iterator(dataset)
reticulate::iter_next(iter)
```

```
## [[1]]
## tf.Tensor([[ 1  2  3  4  5  6  7  8  9 10]], shape=(1, 10), dtype=int32)
##
## [[2]]
## tf.Tensor([11], shape=(1), dtype=int32)
```

Example 3: Temporal regression for many-to-many architectures.

Consider two arrays of scalar values `X` and `Y`,
both of shape `(100,)`. The resulting dataset should consist samples with
20 timestamps each. The samples should not overlap.
To generate a dataset that uses the current timestamp
to predict the corresponding target timestep, you would use:


```r
X <- k_array(1:100)
Y <- X*2

sample_length <- 20
input_dataset <- timeseries_dataset_from_array(
    X, NULL, sequence_length=sample_length, sequence_stride=sample_length)
target_dataset <- timeseries_dataset_from_array(
    Y, NULL, sequence_length=sample_length, sequence_stride=sample_length)


inputs <- reticulate::as_iterator(input_dataset) %>% reticulate::iter_next()
targets <- reticulate::as_iterator(target_dataset) %>% reticulate::iter_next()
```

@param data
array or eager tensor
containing consecutive data points (timesteps).
The first dimension is expected to be the time dimension.

@param targets
Targets corresponding to timesteps in `data`.
`targets[i]` should be the target
corresponding to the window that starts at index `i`
(see example 2 below).
Pass `NULL` if you don't have target data (in this case the dataset
will only yield the input data).

@param sequence_length
Length of the output sequences
(in number of timesteps).

@param sequence_stride
Period between successive output sequences.
For stride `s`, output samples would
start at index `data[i]`, `data[i + s]`, `data[i + 2 * s]`, etc.

@param sampling_rate
Period between successive individual timesteps
within sequences. For rate `r`, timesteps
`data[i], data[i + r], ... data[i + sequence_length]`
are used for creating a sample sequence.

@param batch_size
Number of timeseries samples in each batch
(except maybe the last one). If `NULL`, the data will not be batched
(the dataset will yield individual samples).

@param shuffle
Whether to shuffle output samples,
or instead draw them in chronological order.

@param seed
Optional int; random seed for shuffling.

@param start_index
Optional int; data points earlier (exclusive)
than `start_index` will not be used
in the output sequences. This is useful to reserve part of the
data for test or validation.

@param end_index
Optional int; data points later (exclusive) than `end_index`
will not be used in the output sequences.
This is useful to reserve part of the data for test or validation.

@export
@family utils
@seealso
+ <https:/keras.io/api/data_loading/timeseries#timeseriesdatasetfromarray-function>
+ <https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array>

