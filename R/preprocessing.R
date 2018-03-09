
#' Pads sequences to the same length
#'
#' @details This function transforms a list of `num_samples` sequences (lists
#' of integers) into a matrix of shape `(num_samples, num_timesteps)`.
#' `num_timesteps` is either the `maxlen` argument if provided, or the length
#' of the longest sequence otherwise.
#'
#' Sequences that are shorter than `num_timesteps` are padded with `value` at
#' the end.
#'
#' Sequences longer than `num_timesteps` are truncated so that they fit the
#' desired length. The position where padding or truncation happens is
#' determined by the arguments `padding` and `truncating`, respectively.
#'
#' Pre-padding is the default.
#'
#' @param sequences List of lists where each element is a sequence
#' @param maxlen int, maximum length of all sequences
#' @param dtype type of the output sequences
#' @param padding 'pre' or 'post', pad either before or after each sequence.
#' @param truncating 'pre' or 'post', remove values from sequences larger than
#'   maxlen either in the beginning or in the end of the sequence
#' @param value float, padding value
#'
#' @return Matrix with dimensions (number_of_sequences, maxlen)
#'
#' @family text preprocessing
#'
#' @export
pad_sequences <- function(sequences, maxlen = NULL, dtype = "int32", padding = "pre", 
                          truncating = "pre", value = 0.0) {
  
  # force length-1 sequences to list (so they aren't treated as scalars)
  if (is.list(sequences)) {
    sequences <- lapply(sequences, function(seq) {
      if (length(seq) == 1)
        as.list(seq)
      else
        seq
    })
  }
  
  keras$preprocessing$sequence$pad_sequences(
    sequences = sequences,
    maxlen = as_nullable_integer(maxlen),
    dtype = dtype,
    padding = padding,
    truncating = truncating,
    value = value
  )
}

#' Generates skipgram word pairs.
#' 
#' @details 
#' This function transforms a list of word indexes (lists of integers)
#' into lists of words of the form:
#'
#' - (word, word in the same window), with label 1 (positive samples).
#' - (word, random word from the vocabulary), with label 0 (negative samples).
#'
#' Read more about Skipgram in this gnomic paper by Mikolov et al.:
#' [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)
#' 
#' @param sequence A word sequence (sentence), encoded as a list of word indices
#'   (integers). If using a `sampling_table`, word indices are expected to match
#'   the rank of the words in a reference dataset (e.g. 10 would encode the
#'   10-th most frequently occuring token). Note that index 0 is expected to be
#'   a non-word and will be skipped.
#' @param vocabulary_size Int, maximum possible word index + 1
#' @param window_size Int, size of sampling windows (technically half-window). 
#'   The window of a word `w_i` will be `[i-window_size, i+window_size+1]`
#' @param negative_samples float >= 0. 0 for no negative (i.e. random) samples. 1
#'   for same number as positive samples. 
#' @param shuffle whether to shuffle the word couples before returning them.
#' @param categorical bool. if `FALSE`, labels will be integers (eg. `[0, 1, 1 .. ]`), 
#'   if `TRUE` labels will be categorical eg. `[[1,0],[0,1],[0,1] .. ]`
#' @param sampling_table 1D array of size `vocabulary_size` where the entry i
#'   encodes the probabibily to sample a word of rank i.
#' @param seed Random seed
#'   
#' @return List of `couples`, `labels` where:
#'   - `couples` is a list of 2-element integer vectors: `[word_index, other_word_index]`.
#'   - `labels` is an integer vector of 0 and 1, where 1 indicates that `other_word_index`
#'      was found in the same window as `word_index`, and 0 indicates that `other_word_index`
#'      was random.
#'  - if `categorical` is set to `TRUE`, the labels are categorical, ie. 1 becomes `[0,1]`, 
#'    and 0 becomes `[1, 0]`.
#'   
#' @family text preprocessing   
#'   
#' @export
skipgrams <- function(sequence, vocabulary_size, window_size = 4, negative_samples = 1.0, 
                      shuffle = TRUE, categorical = FALSE, sampling_table = NULL, seed = NULL) {
  
  args <- list(
    sequence = as.integer(sequence),
    vocabulary_size = as.integer(vocabulary_size),
    window_size = as.integer(window_size),
    negative_samples = negative_samples,
    shuffle = shuffle,
    categorical = categorical,
    sampling_table = sampling_table
  )
  
  if (keras_version() >= "2.0.7")
    args$seed <- as_nullable_integer(seed)
  
  sg <- do.call(keras$preprocessing$sequence$skipgrams, args)
  
  sg <- list(
    couples = sg[[1]],
    labels = sg[[2]]
  )
}


#' Generates a word rank-based probabilistic sampling table.
#' 
#' @details 
#' 
#' Used for generating the `sampling_table` argument for [skipgrams()].
#' `sampling_table[[i]]` is the probability of sampling the word i-th most common
#' word in a dataset (more common words should be sampled less frequently, for balance).
#' 
#' The sampling probabilities are generated according to the sampling distribution used in word2vec:
#'
#'  `p(word) = min(1, sqrt(word_frequency / sampling_factor) / (word_frequency / sampling_factor))`
#' 
#' We assume that the word frequencies follow Zipf's law (s=1) to derive a 
#' numerical approximation of frequency(rank):
#' 
#' `frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))`
#' 
#' where `gamma` is the Euler-Mascheroni constant.
#' 
#' @param size Int, number of possible words to sample.
#' @param sampling_factor The sampling factor in the word2vec formula.
#'   
#' @return An array of length `size` where the ith entry is the
#'   probability that a word of rank i should be sampled.
#'   
#' @note The word2vec formula is: p(word) = min(1,
#'   sqrt(word.frequency/sampling_factor) / (word.frequency/sampling_factor))
#'   
#' @family text preprocessing   
#'   
#' @export
make_sampling_table <- function(size, sampling_factor = 1e-05) {
  keras$preprocessing$sequence$make_sampling_table(
    size = as.integer(size),
    sampling_factor = sampling_factor
  )
}

#' Convert text to a sequence of words (or tokens).
#' 
#' @param text Input text (string).
#' @param filters Sequence of characters to filter out.
#' @param lower Whether to convert the input to lowercase.
#' @param split Sentence split marker (string).
#' 
#' @return Words (or tokens)
#' 
#' @family text preprocessing
#' 
#' @export
text_to_word_sequence <- function(text, filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                  lower = TRUE, split=' ') {
  keras$preprocessing$text$text_to_word_sequence(
    text = text,
    filters = filters,
    lower = lower,
    split = split
  )
}

#' One-hot encode a text into a list of word indexes in a vocabulary of size n.
#' 
#' @param n Size of vocabulary (integer)
#'   
#' @inheritParams text_to_word_sequence
#'   
#' @return List of integers in `[1, n]`. Each integer encodes a word (unicity
#'   non-guaranteed).
#'   
#' @family text preprocessing   
#'   
#' @export
text_one_hot <- function(text, n, filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                         lower = TRUE, split = ' ') {
  keras$preprocessing$text$one_hot(
    text = text,
    n = as.integer(n),
    filters = filters,
    lower = lower,
    split = split
  )
}

#' Converts a text to a sequence of indexes in a fixed-size hashing space.
#' 
#' @param text Input text (string).
#' @param n Dimension of the hashing space.
#' @param hash_function if `NULL` uses python `hash` function, can be 'md5' or
#'   any function that takes in input a string and returns a int. Note that
#'   `hash` is not a stable hashing function, so it is not consistent across
#'   different runs, while 'md5' is a stable hashing function.
#' @param filters Sequence of characters to filter out.
#' @param lower Whether to convert the input to lowercase.
#' @param split Sentence split marker (string).
#' 
#' @return  A list of integer word indices (unicity non-guaranteed).
#' 
#' @details 
#' Two or more words may be assigned to the same index, due to possible
#' collisions by the hashing function.
#' 
#' @family text preprocessing   
#'
#' @export
text_hashing_trick <- function(text, n, 
                               hash_function = NULL, 
                               filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
                               lower = TRUE, split = ' ') {
  keras$preprocessing$text$hashing_trick(
    text = text,
    n = as.integer(n),
    hash_function = hash_function,
    filters = filters,
    lower = lower,
    split = split
  )
}



#' Text tokenization utility
#' 
#' Vectorize a text corpus, by turning each text into either a sequence of 
#' integers (each integer being the index of a token in a dictionary) or into a 
#' vector where the coefficient for each token could be binary, based on word 
#' count, based on tf-idf...
#' 
#' @details By default, all punctuation is removed, turning the texts into 
#' space-separated sequences of words (words maybe include the ' character).
#' These sequences are then split into lists of tokens. They will then be
#' indexed or vectorized. `0` is a reserved index that won't be assigned to any
#' word.
#' 
#' @param num_words the maximum number of words to keep, based on word
#'   frequency. Only the most common `num_words` words will be kept.
#' @param filters a string where each element is a character that will be 
#'   filtered from the texts. The default is all punctuation, plus tabs and line
#'   breaks, minus the ' character.
#' @param lower boolean. Whether to convert the texts to lowercase.
#' @param split character or string to use for token splitting.
#' @param char_level if `TRUE`, every character will be treated as a token
#' @param oov_token `NULL` or string If given, it will be added to `word_index``
#'  and used to replace out-of-vocabulary words during text_to_sequence calls.
#'   
#' @section Attributes:
#' The tokenizer object has the following attributes:
#' - `word_counts` --- named list mapping words to the number of times they appeared
#'   on during fit. Only set after `fit_text_tokenizer()` is called on the tokenizer.
#' - `word_docs` --- named list mapping words to the number of documents/texts they
#'    appeared on during fit. Only set after `fit_text_tokenizer()` is called on the tokenizer.
#' - `word_index` --- named list mapping words to their rank/index (int). Only set 
#'    after `fit_text_tokenizer()` is called on the tokenizer.
#' -  `document_count` --- int. Number of documents (texts/sequences) the tokenizer 
#'    was trained on. Only set after `fit_text_tokenizer()` is called on the tokenizer.
#'   
#' @family text tokenization
#'   
#' @export
text_tokenizer <- function(num_words = NULL, filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                           lower = TRUE, split = ' ', char_level = FALSE, oov_token = NULL) {
  args <- list(
    num_words = as_nullable_integer(num_words),
    filters = filters,
    lower = lower,
    split = split,
    char_level = char_level
  )
  
  if (keras_version() >= "2.1.3")
    args$oov_token <- oov_token
  
  do.call(keras$preprocessing$text$Tokenizer, args)
}

#' Update tokenizer internal vocabulary based on a list of texts or list of
#' sequences.
#' 
#' @param object Tokenizer returned by [text_tokenizer()]
#' @param x Vector/list of strings, or a generator of strings (for 
#'   memory-efficiency); Alternatively a list of "sequence" (a sequence is a 
#'   list of integer word indices).
#'
#' @note Required before using [texts_to_sequences()], [texts_to_matrix()], or 
#'   [sequences_to_matrix()].
#'   
#' @family text tokenization
#'   
#' @export
fit_text_tokenizer <- function(object, x) {
  tokenizer <- object
  if (is.list(x))
    tokenizer$fit_on_sequences(x)
  else
    tokenizer$fit_on_texts(x)
  invisible(tokenizer)
}


#' Save a text tokenizer to an external file
#' 
#' Enables persistence of text tokenizers alongside saved models.
#' 
#' @details 
#' You should always use the same text tokenizer for training and  
#' prediction. In many cases however prediction will occur in another
#' session with a version of the model loaded via [load_model_hdf5()].
#' 
#' In this case you need to save the text tokenizer object after training
#' and then reload it prior to prediction.
#' 
#' @param object Text tokenizer fit with [fit_text_tokenizer()]
#' @param filename File to save/load
#' 
#' @family text tokenization
#'
#' @examples \dontrun{
#' 
#' # vectorize texts then save for use in prediction
#' tokenizer <- text_tokenizer(num_words = 10000) %>% 
#' fit_text_tokenizer(tokenizer, texts)
#' save_text_tokenizer(tokenizer, "tokenizer")
#' 
#' # (train model, etc.)
#' 
#' # ...later in another session
#' tokenizer <- load_text_tokenizer("tokenizer")
#' 
#' # (use tokenizer to preprocess data for prediction)
#' 
#' }
#' 
#' @importFrom reticulate py_save_object
#' @export 
save_text_tokenizer <- function(object, filename) {
  py_save_object(object, filename)
  invisible(object)
}


#' @importFrom reticulate py_load_object
#' @rdname save_text_tokenizer
#' @export
load_text_tokenizer <- function(filename) {
  py_load_object(filename)
}






#' Transform each text in texts in a sequence of integers.
#'
#' Only top "num_words" most frequent words will be taken into account.
#' Only words known by the tokenizer will be taken into account.
#' 
#' @param tokenizer Tokenizer
#' @param texts Vector/list of texts (strings).
#' 
#' @family text tokenization
#'   
#' @export
texts_to_sequences <- function(tokenizer, texts) {
  tokenizer$texts_to_sequences(texts)  
}

#' Transforms each text in texts in a sequence of integers.
#' 
#' Only top "num_words" most frequent words will be taken into account.
#' Only words known by the tokenizer will be taken into account.
#' 
#' @inheritParams texts_to_sequences
#' 
#' @return Generator which yields individual sequences
#' 
#' @family text tokenization
#'   
#' @export
texts_to_sequences_generator <- function(tokenizer, texts) {
  tokenizer$texts_to_sequences_generator(texts)
}


#' Convert a list of texts to a matrix.
#'
#' @inheritParams texts_to_sequences
#' 
#' @param mode one of "binary", "count", "tfidf", "freq".
#'  
#' @return A matrix
#'
#' @family text tokenization
#'   
#' @export
texts_to_matrix <- function(tokenizer, texts, mode = c("binary", "count", "tfidf", "freq")) {
  tokenizer$texts_to_matrix(
    texts = texts, 
    mode = mode
  )
}


#' Convert a list of sequences into a matrix.
#'
#' @inheritParams texts_to_matrix
#'
#' @param sequences List of sequences (a sequence is a list of integer word indices).
#'
#' @return A matrix
#'
#' @family text tokenization
#'   
#' @export
sequences_to_matrix <- function(tokenizer, sequences, mode = c("binary", "count", "tfidf", "freq")) {
  
  # force length-1 sequences to list (so they aren't treated as scalars)
  if (is.list(sequences)) {
    sequences <- lapply(sequences, function(seq) {
      if (length(seq) == 1)
        as.list(seq)
      else
        seq
    })
  }
  
  tokenizer$sequences_to_matrix(
    sequences = sequences,
    mode = mode
  )
}


#' Loads an image into PIL format.
#'
#' @param path Path to image file
#' @param grayscale Boolean, whether to load the image as grayscale.
#' @param target_size Either `NULL` (default to original size) or integer vector
#'   `(img_height, img_width)`.
#' @param interpolation Interpolation method used to resample the image if the
#'   target size is different from that of the loaded image. Supported methods
#'   are "nearest", "bilinear", and "bicubic". If PIL version 1.1.3 or newer is
#'   installed, "lanczos" is also supported. If PIL version 3.4.0 or newer is
#'   installed, "box" and "hamming" are also supported. By default, "nearest"
#'   is used.
#'
#' @return A PIL Image instance.
#'
#' @family image preprocessing
#'
#' @export
image_load <- function(path, grayscale = FALSE, target_size = NULL,
                       interpolation = "nearest") {
  
  if (!have_pillow())
    stop("The Pillow Python package is required to load images")
  
  # normalize target_size
  if (!is.null(target_size)) {
    if (length(target_size) != 2)
      stop("target_size must be 2 element integer vector")
    target_size <- as.integer(target_size)
    target_size <- tuple(target_size[[1]], target_size[[2]])
  }
  
  args <- list(
    path = normalize_path(path),
    grayscale = grayscale,
    target_size = target_size
  )
  
  if (keras_version() >= "2.0.9")
    args$interpolation <- interpolation
  
  do.call(keras$preprocessing$image$load_img, args)
}



#' 3D array representation of images
#' 
#' 3D array that represents an image with dimensions (height,width,channels) or 
#'  (channels,height,width) depending on the data_format.
#' 
#' @param img Image
#' @param path Path to save image to
#' @param width Width to resize to
#' @param height Height to resize to
#' @param data_format Image data format ("channels_last" or "channels_first")
#' 
#' @family image preprocessing
#' 
#' @export
image_to_array <- function(img, data_format = c("channels_last", "channels_first")) {
  keras$preprocessing$image$img_to_array(
    img = img,
    data_format = match.arg(data_format)
  )
}

#' @rdname image_to_array
#' @export
image_array_resize <- function(img, height, width,
                               data_format = c("channels_last", "channels_first")) {
  
  # imports
  np <- import("numpy")
  scipy <- import("scipy")
  
  # make copy as necessary
  img <- np$copy(img)
  
  # capture dimensions and reduce to 3 if necessary
  dims <- dim(img)
  is_4d_array <- FALSE
  if (length(dims) == 4 && dims[[1]] == 1) {
    is_4d_array <- TRUE
    img <- array_reshape(img, dims[-1])
  }
  
  # calculate zoom factors (invert the dimensions to reflect height,width
  # order of numpy/scipy array represenations of images)
  data_format <- match.arg(data_format)
  if (data_format == "channels_last") {
    factors <- tuple(
      height / dim(img)[[1]],
      width / dim(img)[[2]],
      1
    )
  } else {
    factors <- tuple(
      1,
      height / dim(img)[[1]],
      width / dim(img)[[2]],
    )
  }
  
  # zoom
  img <- scipy$ndimage$zoom(img, factors, order = 1L)
  
  # reshape if necessary
  if (is_4d_array)
    img <- array_reshape(img, dim = c(1, dim(img)))
  
  # return
  img
}

#' @rdname image_to_array
#' @export
image_array_save <- function(img, path) {
  scipy <- import("scipy")
  invisible(scipy$misc$imsave(path, img))
}




#' Generate minibatches of image data with real-time data augmentation.
#' 
#' @param featurewise_center set input mean to 0 over the dataset.
#' @param samplewise_center set each sample mean to 0.
#' @param featurewise_std_normalization divide inputs by std of the dataset.
#' @param samplewise_std_normalization divide each input by its std.
#' @param zca_whitening apply ZCA whitening.
#' @param zca_epsilon Epsilon for ZCA whitening. Default is 1e-6.
#' @param rotation_range degrees (0 to 180).
#' @param width_shift_range fraction of total width.
#' @param height_shift_range fraction of total height.
#' @param brightness_range the range of brightness to apply
#' @param shear_range shear intensity (shear angle in radians).
#' @param zoom_range amount of zoom. if scalar z, zoom will be randomly picked
#'   in the range `[1-z, 1+z]`. A sequence of two can be passed instead to select
#'   this range.
#' @param channel_shift_range shift range for each channels.
#' @param fill_mode One of "constant", "nearest", "reflect" or "wrap".  
#' Points outside the boundaries of the input are filled according to 
#' the given mode:
#'    - "constant": `kkkkkkkk|abcd|kkkkkkkk` (`cval=k`)
#'    - "nearest":  `aaaaaaaa|abcd|dddddddd`
#'    - "reflect":  `abcddcba|abcd|dcbaabcd`
#'    - "wrap":     `abcdabcd|abcd|abcdabcd`
#' @param cval value used for points outside the boundaries when fill_mode is
#'   'constant'. Default is 0.
#' @param horizontal_flip whether to randomly flip images horizontally.
#' @param vertical_flip whether to randomly flip images vertically.
#' @param rescale rescaling factor. If NULL or 0, no rescaling is applied,
#'   otherwise we multiply the data by the value provided (before applying any
#'   other transformation).
#' @param preprocessing_function function that will be implied on each input.
#'   The function will run before any other modification on it. The function
#'   should take one argument: one image (tensor with rank 3), and should
#'   output a tensor with the same shape.
#' @param data_format 'channels_first' or 'channels_last'. In 'channels_first'
#'   mode, the channels dimension (the depth) is at index 1, in 'channels_last'
#'   mode it is at index 3. It defaults to the `image_data_format` value found
#'   in your Keras config file at `~/.keras/keras.json`. If you never set it,
#'   then it will be "channels_last".
#' @param validation_split fraction of images reserved for validation (strictly between 0 and 1).
#'   
#' @export
image_data_generator <- function(featurewise_center = FALSE, samplewise_center = FALSE, 
                                 featurewise_std_normalization = FALSE, samplewise_std_normalization = FALSE, 
                                 zca_whitening = FALSE, zca_epsilon = 1e-6, rotation_range = 0.0, width_shift_range = 0.0, 
                                 height_shift_range = 0.0, brightness_range = NULL, shear_range = 0.0, zoom_range = 0.0, channel_shift_range = 0.0, 
                                 fill_mode = "nearest", cval = 0.0, horizontal_flip = FALSE, vertical_flip = FALSE, 
                                 rescale = NULL, preprocessing_function = NULL, data_format = NULL, validation_split=0.0) {
  args <- list(
    featurewise_center = featurewise_center,
    samplewise_center = samplewise_center,
    featurewise_std_normalization = featurewise_std_normalization,
    samplewise_std_normalization = samplewise_std_normalization,
    zca_whitening = zca_whitening,
    rotation_range = rotation_range,
    width_shift_range = width_shift_range,
    height_shift_range = height_shift_range,
    shear_range = shear_range,
    zoom_range = zoom_range,
    channel_shift_range = channel_shift_range,
    fill_mode = fill_mode,
    cval = cval,
    horizontal_flip = horizontal_flip,
    vertical_flip = vertical_flip,
    rescale = rescale,
    preprocessing_function = preprocessing_function,
    data_format = data_format
  )
  if (keras_version() >= "2.0.4")
    args$zca_epsilon <- zca_epsilon
  if (keras_version() >= "2.1.5") {
    args$brightness_range <- brightness_range
    args$validation_split <- validation_split
  }
  
  do.call(keras$preprocessing$image$ImageDataGenerator, args)
  
}


#' Retrieve the next item from a generator
#' 
#' Use to retrieve items from generators (e.g. [image_data_generator()]). Will return
#' either the next item or `NULL` if there are no more items.
#' 
#' @param generator Generator 
#' @param completed Sentinel value to return from `generator_next()` if the iteration
#'   completes (defaults to `NULL` but can be any R value you specify).
#'
#' @export
generator_next <- function(generator, completed = NULL) {
  reticulate::iter_next(generator, completed = completed)
}


#' Fit image data generator internal statistics to some sample data.
#' 
#' Required for `featurewise_center`, `featurewise_std_normalization`
#' and `zca_whitening`.
#' 
#' @param object [image_data_generator()]
#' @param x  array, the data to fit on (should have rank 4). In case of grayscale data,
#' the channels axis should have value 1, and in case of RGB data, it should have value 3.
#' @param augment Whether to fit on randomly augmented samples
#' @param rounds If `augment`, how many augmentation passes to do over the data
#' @param seed random seed.
#' 
#' @family image preprocessing
#' 
#' @export
fit_image_data_generator <- function(object, x, augment = FALSE, rounds = 1, seed = NULL) {
  generator <- object
  history <- generator$fit(
    x = keras_array(x),
    augment = augment,
    rounds = as.integer(rounds),
    seed = seed
  )
  invisible(history)
}

#' Generates batches of augmented/normalized data from image data and labels
#' 
#' @details Yields batches indefinitely, in an infinite loop.
#'   
#' @param generator Image data generator to use for augmenting/normalizing image
#'   data.
#' @param x data. Should have rank 4. In case of grayscale data, the channels 
#'   axis should have value 1, and in case of RGB data, it should have value 3.
#' @param y labels (can be `NULL` if no labels are required)
#' @param batch_size int (default: `32`).
#' @param shuffle boolean (defaut: `TRUE`).
#' @param seed int (default: `NULL`).
#' @param save_to_dir `NULL` or str (default: `NULL`). This allows you to 
#'   optionally specify a directory to which to save the augmented pictures being
#'   generated (useful for visualizing what you are doing).
#' @param save_prefix str (default: ''). Prefix to use for filenames of saved 
#'   pictures (only relevant if `save_to_dir` is set).
#' @param save_format one of "png", "jpeg" (only relevant if save_to_dir is 
#'   set). Default: "png".
#' @param subset Subset of data (`"training"` or `"validation"`) if
#'   `validation_split` is set in [image_data_generator()].
#'   
#' @section Yields: `(x, y)` where `x` is an array of image data and `y` is a 
#'   array of corresponding labels. The generator loops indefinitely.
#'   
#' @family image preprocessing
#'   
#' @export
flow_images_from_data <- function(
  x, y = NULL, generator = image_data_generator(), batch_size = 32, 
  shuffle = TRUE, seed = NULL, 
  save_to_dir = NULL, save_prefix = "", save_format = 'png', subset = NULL) {
  
  args <- list(
    x = keras_array(x),
    y = keras_array(y),
    batch_size = as.integer(batch_size),
    shuffle = shuffle,
    seed = as_nullable_integer(seed),
    save_to_dir = normalize_path(save_to_dir),
    save_prefix = save_prefix,
    save_format = save_format
  )
  
  if (keras_version() >= "2.1.5")
    args$subset <- subset
  
  do.call(generator$flow, args)
}

#' Generates batches of data from images in a directory (with optional
#' augmented/normalized data)
#'
#' @details Yields batches indefinitely, in an infinite loop.
#'
#' @inheritParams image_load
#' @inheritParams flow_images_from_data
#'
#' @param generator Image data generator (default generator does no data
#'   augmentation/normalization transformations)
#' @param directory path to the target directory. It should contain one
#'   subdirectory per class. Any PNG, JPG, BMP, PPM, or TIF images inside each
#'   of the subdirectories directory tree will be included in the generator.
#'   See [this script](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
#'   for more details.
#' @param target_size integer vectir, default: `c(256, 256)`. The dimensions to
#'   which all images found will be resized.
#' @param color_mode one of "grayscale", "rbg". Default: "rgb". Whether the
#'   images will be converted to have 1 or 3 color channels.
#' @param classes optional list of class subdirectories (e.g. `c('dogs',
#'   'cats')`). Default: `NULL`, If not provided, the list of classes will be
#'   automatically inferred (and the order of the classes, which will map to
#'   the label indices, will be alphanumeric).
#' @param class_mode one of "categorical", "binary", "sparse" or `NULL`.
#'   Default: "categorical". Determines the type of label arrays that are
#'   returned: "categorical" will be 2D one-hot encoded labels, "binary" will
#'   be 1D binary labels, "sparse" will be 1D integer labels. If `NULL`, no
#'   labels are returned (the generator will only yield batches of image data,
#'   which is useful to use [predict_generator()], [evaluate_generator()],
#'   etc.).
#' @param follow_links whether to follow symlinks inside class subdirectories
#'   (default: `FALSE`)
#'
#' @section Yields: `(x, y)` where `x` is an array of image data and `y` is a
#'   array of corresponding labels. The generator loops indefinitely.
#'
#' @family image preprocessing
#'
#' @export
flow_images_from_directory <- function(
  directory, generator = image_data_generator(), target_size = c(256, 256), color_mode = "rgb",
  classes = NULL, class_mode = "categorical",
  batch_size = 32, shuffle = TRUE, seed = NULL,
  save_to_dir = NULL, save_prefix = "", save_format = "png",
  follow_links = FALSE, subset = NULL, interpolation = "nearest") {
  
  args <- list(
    directory = normalize_path(directory),
    target_size = as.integer(target_size),
    color_mode = color_mode,
    classes = classes,
    class_mode = class_mode,
    batch_size = as.integer(batch_size),
    shuffle = shuffle,
    seed = as_nullable_integer(seed),
    save_to_dir = normalize_path(save_to_dir),
    save_prefix = save_prefix,
    save_format = save_format,
    follow_links = follow_links
  )
  
  if (keras_version() >= "2.1.2")
    args$interpolation <- interpolation
  
  if (keras_version() >= "2.1.5")
    args$subset <- subset
  
  do.call(generator$flow_from_directory, args)
}





