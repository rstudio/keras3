# IMDB Movie reviews sentiment classification

Dataset of 25,000 movies reviews from IMDB, labeled by sentiment
(positive/negative). Reviews have been preprocessed, and each review is
encoded as a sequence of word indexes (integers). For convenience, words
are indexed by overall frequency in the dataset, so that for instance
the integer "3" encodes the 3rd most frequent word in the data. This
allows for quick filtering operations such as: "only consider the top
10,000 most common words, but eliminate the top 20 most common words".

## Usage

``` r
dataset_imdb(
  path = "imdb.npz",
  num_words = NULL,
  skip_top = 0L,
  maxlen = NULL,
  seed = 113L,
  start_char = 1L,
  oov_char = 2L,
  index_from = 3L,
  convert = TRUE
)

dataset_imdb_word_index(path = "imdb_word_index.json")
```

## Arguments

- path:

  Where to cache the data (relative to `~/.keras/dataset`).

- num_words:

  Max number of words to include. Words are ranked by how often they
  occur (in the training set) and only the most frequent words are kept

- skip_top:

  Skip the top N most frequently occuring words (which may not be
  informative).

- maxlen:

  sequences longer than this will be filtered out.

- seed:

  random seed for sample shuffling.

- start_char:

  The start of a sequence will be marked with this character. Set to 1
  because 0 is usually the padding character.

- oov_char:

  Words that were cut out because of the `num_words` or `skip_top` limit
  will be replaced with this character.

- index_from:

  Index actual words with this index and higher.

- convert:

  When `TRUE` (default) the datasets are returned as R arrays. If
  `FALSE`, objects are returned as NumPy arrays.

## Value

Lists of training and test data: `train$x, train$y, test$x, test$y`.

    train/
      - x
      - y
    test/
      - x
      - y

The `x` data includes integer sequences. If the `num_words` argument was
specific, the maximum possible index value is `num_words-1`. If the
`maxlen` argument was specified, the largest possible sequence length is
`maxlen`.

The `y` data includes a set of integer labels (0 or 1).

    str(dataset_imdb())

    ## List of 2
    ##  $ train:List of 2
    ##   ..$ x:List of 25000
    ##   .. ..$ : int [1:218] 1 14 22 16 43 530 973 1622 1385 65 ...
    ##   .. ..$ : int [1:189] 1 194 1153 194 8255 78 228 5 6 1463 ...
    ##   .. ..$ : int [1:141] 1 14 47 8 30 31 7 4 249 108 ...
    ##   .. ..$ : int [1:550] 1 4 18609 16085 33 2804 4 2040 432 111 ...
    ##   .. ..$ : int [1:147] 1 249 1323 7 61 113 10 10 13 1637 ...
    ##   .. ..$ : int [1:43] 1 778 128 74 12 630 163 15 4 1766 ...
    ##   .. .. [list output truncated]
    ##   ..$ y: int [1:25000] 1 0 0 1 0 0 1 0 1 0 ...
    ##  $ test :List of 2
    ##   ..$ x:List of 25000
    ##   .. ..$ : int [1:68] 1 591 202 14 31 6 717 10 10 18142 ...
    ##   .. ..$ : int [1:260] 1 14 22 3443 6 176 7 5063 88 12 ...
    ##   .. ..$ : int [1:603] 1 111 748 4368 1133 33782 24563 4 87 1551 ...
    ##   .. ..$ : int [1:181] 1 13 1228 119 14 552 7 20 190 14 ...
    ##   .. ..$ : int [1:108] 1 40 49 85 84 1040 146 6 783 254 ...
    ##   .. ..$ : int [1:132] 1 146 427 5718 14 20 218 112 2962 32 ...
    ##   .. .. [list output truncated]
    ##   ..$ y: int [1:25000] 0 1 1 0 1 1 1 0 0 1 ...

    str(dataset_imdb(convert = FALSE))

    ## List of 2
    ##  $ train:List of 2
    ##   ..$ x: <numpy.ndarray shape(25000), dtype=object>
    ##   ..$ y: <numpy.ndarray shape(25000), dtype=int64>
    ##  $ test :List of 2
    ##   ..$ x: <numpy.ndarray shape(25000), dtype=object>
    ##   ..$ y: <numpy.ndarray shape(25000), dtype=int64>

The `dataset_imdb_word_index()` function returns a list where the names
are words and the values are integer.

## Details

As a convention, "0" does not stand for a specific word, but instead is
used to encode any unknown word.

## See also

Other datasets:  
[`dataset_boston_housing()`](https://keras3.posit.co/reference/dataset_boston_housing.md)  
[`dataset_california_housing()`](https://keras3.posit.co/reference/dataset_california_housing.md)  
[`dataset_cifar10()`](https://keras3.posit.co/reference/dataset_cifar10.md)  
[`dataset_cifar100()`](https://keras3.posit.co/reference/dataset_cifar100.md)  
[`dataset_fashion_mnist()`](https://keras3.posit.co/reference/dataset_fashion_mnist.md)  
[`dataset_mnist()`](https://keras3.posit.co/reference/dataset_mnist.md)  
[`dataset_reuters()`](https://keras3.posit.co/reference/dataset_reuters.md)  
