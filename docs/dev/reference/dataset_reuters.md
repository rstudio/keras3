# Reuters newswire topics classification

Dataset of 11,228 newswires from Reuters, labeled over 46 topics. As
with
[`dataset_imdb()`](https://keras3.posit.co/dev/reference/dataset_imdb.md)
, each wire is encoded as a sequence of word indexes (same conventions).

## Usage

``` r
dataset_reuters(
  path = "reuters.npz",
  num_words = NULL,
  skip_top = 0L,
  maxlen = NULL,
  test_split = 0.2,
  seed = 113L,
  start_char = 1L,
  oov_char = 2L,
  index_from = 3L,
  convert = TRUE
)

dataset_reuters_word_index(path = "reuters_word_index.pkl")
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

  Truncate sequences after this length.

- test_split:

  Fraction of the dataset to be used as test data.

- seed:

  Random seed for sample shuffling.

- start_char:

  The start of a sequence will be marked with this character. Set to 1
  because 0 is usually the padding character.

- oov_char:

  words that were cut out because of the `num_words` or `skip_top` limit
  will be replaced with this character.

- index_from:

  index actual words with this index and higher.

- convert:

  When `TRUE` (default) the datasets are returned as R arrays. If
  `FALSE`, objects are returned as NumPy arrays.

## Value

Lists of training and test data: `train$x, train$y, test$x, test$y` with
same format as
[`dataset_imdb()`](https://keras3.posit.co/dev/reference/dataset_imdb.md).
The `dataset_reuters_word_index()` function returns a list where the
names are words and the values are integer. e.g.
`word_index[["giraffe"]]` might return `1234`.

    train/
    ├─ x
    └─ y
    test/
    ├─ x
    └─ y

    str(dataset_reuters())

    ## List of 2
    ##  $ train:List of 2
    ##   ..$ x:List of 8982
    ##   .. ..$ : int [1:87] 1 27595 28842 8 43 10 447 5 25 207 ...
    ##   .. ..$ : int [1:56] 1 3267 699 3434 2295 56 16784 7511 9 56 ...
    ##   .. ..$ : int [1:139] 1 53 12 284 15 14 272 26 53 959 ...
    ##   .. ..$ : int [1:224] 1 4 686 867 558 4 37 38 309 2276 ...
    ##   .. ..$ : int [1:101] 1 8295 111 8 25 166 40 638 10 436 ...
    ##   .. ..$ : int [1:116] 1 4 37 38 309 213 349 1632 48 193 ...
    ##   .. .. [list output truncated]
    ##   ..$ y: int [1:8982] 3 4 3 4 4 4 4 3 3 16 ...
    ##  $ test :List of 2
    ##   ..$ x:List of 2246
    ##   .. ..$ : int [1:145] 1 4 1378 2025 9 697 4622 111 8 25 ...
    ##   .. ..$ : int [1:745] 1 2768 283 122 7 4 89 544 463 29 ...
    ##   .. ..$ : int [1:228] 1 4 309 2276 4759 5 2015 403 1920 33 ...
    ##   .. ..$ : int [1:172] 1 11786 13716 65 9 249 1096 8 16 515 ...
    ##   .. ..$ : int [1:187] 1 470 354 18270 4231 62 2373 509 1687 5138 ...
    ##   .. ..$ : int [1:80] 1 53 134 26 14 102 26 39 5150 18 ...
    ##   .. .. [list output truncated]
    ##   ..$ y: int [1:2246] 3 10 1 4 4 3 3 3 3 3 ...

    str(dataset_reuters(convert = FALSE))

    ## List of 2
    ##  $ train:List of 2
    ##   ..$ x: <numpy.ndarray shape(8982), dtype=object>
    ##   ..$ y: <numpy.ndarray shape(8982), dtype=int64>
    ##  $ test :List of 2
    ##   ..$ x: <numpy.ndarray shape(2246), dtype=object>
    ##   ..$ y: <numpy.ndarray shape(2246), dtype=int64>

## See also

Other datasets:  
[`dataset_boston_housing()`](https://keras3.posit.co/dev/reference/dataset_boston_housing.md)  
[`dataset_california_housing()`](https://keras3.posit.co/dev/reference/dataset_california_housing.md)  
[`dataset_cifar10()`](https://keras3.posit.co/dev/reference/dataset_cifar10.md)  
[`dataset_cifar100()`](https://keras3.posit.co/dev/reference/dataset_cifar100.md)  
[`dataset_fashion_mnist()`](https://keras3.posit.co/dev/reference/dataset_fashion_mnist.md)  
[`dataset_imdb()`](https://keras3.posit.co/dev/reference/dataset_imdb.md)  
[`dataset_mnist()`](https://keras3.posit.co/dev/reference/dataset_mnist.md)  
