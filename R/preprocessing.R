
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
#' [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781v3.pdf)
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
#' @param filters Sequence of characters to filter out such as
#'   punctuation. Default includes basic punctuation, tabs, and newlines.
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
#' @param input_text Input text (string).
#' @inheritParams text_to_word_sequence
#' @param text for compatibility purpose. use `input_text` instead.
#'
#' @return List of integers in `[1, n]`. Each integer encodes a word (unicity
#'   non-guaranteed).
#'
#' @family text preprocessing
#'
#' @export
text_one_hot <- function(input_text, n, filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                         lower = TRUE, split = ' ', text = NULL) {

  if (tensorflow::tf_version() >= "2.3" && !is.null(text)) {
    warning("text is deprecated as of TF 2.3. use input_text instead")
    if (!missing(input_text))
      stop("input_text and text must not be bopth specified")
    input_text <- text
  }

  keras$preprocessing$text$one_hot(
    input_text,
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
#' @param hash_function if `NULL` uses the Python `hash()` function. Otherwise can be `'md5'` or
#'   any function that takes in input a string and returns an int. Note that
#'   `hash` is not a stable hashing function, so it is not consistent across
#'   different runs, while `'md5'` is a stable hashing function.
#' @param filters Sequence of characters to filter out such as
#'   punctuation. Default includes basic punctuation, tabs, and newlines.
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
  if (length(text) != 1) {
    stop("`text` should be length 1.")
  }
  if (is.na(text)) {
    return(NA_integer_)
  }
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
  else {
    tokenizer$fit_on_texts(if (is.function(x)) reticulate::py_iterator(x) else as_texts(x))
  }
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
  tokenizer$texts_to_sequences(as_texts(texts))
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
  tokenizer$texts_to_sequences_generator(as_texts(texts))
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
    texts = as_texts(texts),
    mode = mode
  )
}

as_texts <- function(texts) {
  if (is.character(texts) && length(texts) == 1)
    as.array(texts)
  else
    texts
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
#' @param grayscale DEPRECATED use `color_mode="grayscale"`
#' @param color_mode One of `{"grayscale", "rgb", "rgba"}`.
#'    Default: `"rgb"`. The desired image format.
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
image_load <- function(path, grayscale = FALSE, color_mode='rgb',
                       target_size = NULL,
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
    color_mode = color_mode,
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
#' (channels,height,width) depending on the data_format.
#'
#' @param img Image
#' @param path Path to save image to
#' @param width Width to resize to
#' @param height Height to resize to
#' @param data_format Image data format ("channels_last" or "channels_first")
#' @param file_format Optional file format override. If omitted, the format to
#'   use is determined from the filename extension. If a file object was used
#'   instead of a filename, this parameter should always be used.
#' @param scale Whether to rescale image values to be within 0,255
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
image_array_save <- function(img, path, data_format = NULL, file_format = NULL, scale = TRUE) {
  if (keras_version() >= "2.2.0") {
    keras$preprocessing$image$save_img(
      path, img,
      data_format = data_format,
      file_format = file_format,
      scale = scale
    )
  } else {
    pil <- import("PIL")
    pil$Image$fromarray(reticulate::r_to_py(img)$astype("uint8"))$save(path)
  }
}




#' [Deprecated] Generate batches of image data with real-time data augmentation.
#' The data will be looped over (in batches).
#'
#' Deprecated: `image_data_generator` is not
#' recommended for new code. Prefer loading images with
#' `image_dataset_from_directory` and transforming the output
#' TF Dataset with preprocessing layers. For more information, see the
#' tutorials for loading images and augmenting images, as well as the
#' preprocessing layer guide.
#'
#' @param featurewise_center Set input mean to 0 over the dataset, feature-wise.
#' @param samplewise_center Boolean. Set each sample mean to 0.
#' @param featurewise_std_normalization Divide inputs by std of the dataset,
#'   feature-wise.
#' @param samplewise_std_normalization Divide each input by its std.
#' @param zca_whitening apply ZCA whitening.
#' @param zca_epsilon Epsilon for ZCA whitening. Default is 1e-6.
#' @param rotation_range degrees (0 to 180).
#' @param width_shift_range fraction of total width.
#' @param height_shift_range fraction of total height.
#' @param brightness_range the range of brightness to apply
#' @param shear_range shear intensity (shear angle in radians).
#' @param zoom_range amount of zoom. if scalar z, zoom will be randomly picked
#'   in the range `[1-z, 1+z]`. A sequence of two can be passed instead to
#'   select this range.
#' @param channel_shift_range shift range for each channels.
#' @param fill_mode One of "constant", "nearest", "reflect" or "wrap". Points
#'   outside the boundaries of the input are filled according to the given mode:
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
#'   should take one argument: one image (tensor with rank 3), and should output
#'   a tensor with the same shape.
#' @param data_format 'channels_first' or 'channels_last'. In 'channels_first'
#'   mode, the channels dimension (the depth) is at index 1, in 'channels_last'
#'   mode it is at index 3. It defaults to the `image_data_format` value found
#'   in your Keras config file at `~/.keras/keras.json`. If you never set it,
#'   then it will be "channels_last".
#' @param validation_split fraction of images reserved for validation (strictly
#'   between 0 and 1).
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

  if(is.function(preprocessing_function) &&
     !inherits(preprocessing_function, "python.builtin.object"))
    args$preprocessing_function <-
      reticulate::py_main_thread_func(preprocessing_function)

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
#' @param sample_weight Sample weights.
#'
#' @section Yields: `(x, y)` where `x` is an array of image data and `y` is a
#'   array of corresponding labels. The generator loops indefinitely.
#'
#' @family image preprocessing
#'
#' @importFrom reticulate iter_next
#' @export
flow_images_from_data <- function(
  x, y = NULL, generator = image_data_generator(), batch_size = 32,
  shuffle = TRUE, sample_weight = NULL, seed = NULL,
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
  stopifnot(args$batch_size > 0)

  if (keras_version() >= "2.1.5")
    args$subset <- subset

  if (keras_version() >= "2.2.0")
    args$sample_weight <- sample_weight

  iterator <- do.call(generator$flow, args)

  if(!is.null(generator$preprocessing_function)) {
    # user supplied a custom preprocessing function, which likely is an R
    # function that must be called from the main thread. Wrap this in
    # py_iterator(prefetch=1) to ensure we don't end in a deadlock.
    iter_env <- new.env(parent = parent.env(environment())) # pkg namespace
    iter_env$.iterator <- iterator
    expr <- substitute(py_iterator(function() iter_next(iterator), prefetch=1L),
                       list(iterator = quote(.iterator)))
    iterator <- eval(expr, iter_env)
  }

  iterator

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
#' @param target_size integer vector, default: `c(256, 256)`. The dimensions to
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
  stopifnot(args$batch_size > 0)

  if (keras_version() >= "2.1.2")
    args$interpolation <- interpolation

  if (keras_version() >= "2.1.5")
    args$subset <- subset

  do.call(generator$flow_from_directory, args)
}

#' Takes the dataframe and the path to a directory and generates batches of
#' augmented/normalized data.
#'
#' @details Yields batches indefinitely, in an infinite loop.
#'
#' @inheritParams image_load
#' @inheritParams flow_images_from_data
#'
#' @param dataframe `data.frame` containing the filepaths relative to
#'   directory (or absolute paths if directory is `NULL`) of the images in a
#'   character column. It should include other column/s depending on the
#'   `class_mode`:
#'   - if `class_mode` is "categorical" (default value) it must
#'   include the `y_col` column with the class/es of each image. Values in
#'   column can be character/list if a single class or list if multiple classes.
#'   - if `class_mode` is "binary" or "sparse" it must include the given
#'   `y_col` column with class values as strings.
#'   - if `class_mode` is "other" it
#'   should contain the columns specified in `y_col`.
#'   - if `class_mode` is "input" or NULL no extra column is needed.
#' @param directory character, path to the directory to read images from.
#'   If `NULL`, data in `x_col` column should be absolute paths.
#' @param x_col character, column in dataframe that contains the filenames
#'   (or absolute paths if directory is `NULL`).
#' @param y_col string or list, column/s in dataframe that has the target data.
#' @param color_mode one of "grayscale", "rgb". Default: "rgb". Whether the
#'   images will be converted to have 1 or 3 color channels.
#' @param drop_duplicates (deprecated in TF >= 2.3) Boolean, whether to drop
#'   duplicate rows based on filename. The default value is `TRUE`.
#' @param classes optional list of classes (e.g. `c('dogs', 'cats')`. Default:
#'  `NULL` If not provided, the list of classes will be automatically inferred
#'  from the `y_col`, which will map to the label indices, will be alphanumeric).
#'  The dictionary containing the mapping from class names to class indices
#'  can be obtained via the attribute `class_indices`.
#' @param class_mode one of "categorical", "binary", "sparse", "input", "other" or None.
#'   Default: "categorical". Mode for yielding the targets:
#'   * "binary": 1D array of binary labels,
#'   * "categorical": 2D array of one-hot encoded labels. Supports multi-label output.
#'   * "sparse": 1D array of integer labels,
#'   * "input": images identical to input images (mainly used to work with autoencoders),
#'   * "other": array of y_col data,
#'   * "multi_output": allow to train a multi-output model. Y is a list or a vector.
#'   `NULL`, no targets are returned (the generator will only yield batches of
#'   image data, which is useful to use in  `predict_generator()`).
#'
#' @note
#' This functions requires that `pandas` (Python module) is installed in the
#' same environment as `tensorflow` and `keras`.
#'
#' If you are using `r-tensorflow` (the default environment) you can install
#' `pandas` by running `reticulate::virtualenv_install("pandas", envname = "r-tensorflow")`
#' or `reticulate::conda_install("pandas", envname = "r-tensorflow")` depending on
#' the kind of environment you are using.
#'
#' @section Yields: `(x, y)` where `x` is an array of image data and `y` is a
#'   array of corresponding labels. The generator loops indefinitely.
#'
#' @family image preprocessing
#' @export
flow_images_from_dataframe <- function(
  dataframe, directory = NULL, x_col = "filename", y_col = "class",
  generator = image_data_generator(), target_size = c(256,256),
  color_mode = "rgb", classes = NULL, class_mode = "categorical",
  batch_size = 32, shuffle = TRUE, seed = NULL, save_to_dir = NULL,
  save_prefix = "", save_format = "png", subset = NULL,
  interpolation = "nearest", drop_duplicates = NULL) {

  if (!reticulate::py_module_available("pandas"))
    stop("Pandas (Python module) must be installed in the same environment as Keras.",
         'Install it using reticulate::virtualenv_install("pandas", envname = "r-tensorflow") ',
         'or reticulate::conda_install("pandas", envname = "r-tensorflow") depending on ',
         'the kind of environment you are using.')

  args <- list(
    dataframe = as.data.frame(dataframe),
    directory = normalize_path(directory),
    x_col = x_col, y_col = y_col,
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
    drop_duplicates = drop_duplicates
  )
  stopifnot(args$batch_size > 0)

  if (keras_version() >= "2.1.2")
    args$interpolation <- interpolation

  if (keras_version() >= "2.1.5")
    args$subset <- subset

  if(!is.null(drop_duplicates) && tensorflow::tf_version() >= "2.3") {
    warning("\'drop_duplicates\' is deprecated as of tensorflow 2.3 and will be ignored. Make sure the supplied dataframe does not contain duplicates.")
    args$drop_duplicates <- NULL
  }

  if (is.null(drop_duplicates) && tensorflow::tf_version() < "2.3")
    args$drop_duplicates <- TRUE

  do.call(generator$flow_from_dataframe, args)
}

#' Create a dataset from a directory
#'
#' Generates a `tf.data.Dataset` from image files in a directory.
#'
#' If your directory structure is:
#'
#' ````
#' main_directory/
#' ...class_a/
#' ......a_image_1.jpg
#' ......a_image_2.jpg
#' ...class_b/
#' ......b_image_1.jpg
#' ......b_image_2.jpg
#' ````
#'
#' Then calling `image_dataset_from_directory(main_directory, labels='inferred')`
#' will return a `tf.data.Dataset` that yields batches of images from the
#' subdirectories class_a and class_b, together with labels 0 and 1 (0
#' corresponding to class_a and 1 corresponding to class_b).
#'
#' Supported image formats: jpeg, png, bmp, gif. Animated gifs are truncated to
#' the first frame.
#'
#' @param directory Directory where the data is located. If labels is
#'   "inferred", it should contain subdirectories, each containing images for a
#'   class. Otherwise, the directory structure is ignored.
#' @param labels Either "inferred" (labels are generated from the directory
#'   structure), or a list/tuple of integer labels of the same size as the
#'   number of image files found in the directory. Labels should be sorted
#'   according to the alphanumeric order of the image file paths (obtained via
#'   os.walk(directory) in Python).
#' @param label_mode Valid values:
#'
#'   - 'int': labels are encoded as integers (e.g.
#'   for sparse_categorical_crossentropy loss).
#'
#'   - 'categorical': labels are encoded as a categorical vector (e.g. for
#'   categorical_crossentropy loss).
#'
#'   - 'binary': labels (there can be only 2) are encoded as float32 scalars
#'   with values 0 or 1 (e.g. for binary_crossentropy).
#'
#'   - `NULL`: (no labels).
#' @param class_names Only valid if "labels" is "inferred". This is the explict
#'   list of class names (must match names of subdirectories). Used to control
#'   the order of the classes (otherwise alphanumerical order is used).
#' @param color_mode One of "grayscale", "rgb", "rgba". Default: "rgb". Whether
#'   the images will be converted to have 1, 3, or 4 channels.
#' @param batch_size Size of the batches of data. Default: 32.
#' @param image_size Size to resize images to after they are read from disk.
#'   Defaults to (256, 256). Since the pipeline processes batches of images that
#'   must all have the same size, this must be provided.
#' @param shuffle Whether to shuffle the data. Default: TRUE. If set to FALSE,
#'   sorts the data in alphanumeric order.
#' @param seed Optional random seed for shuffling and transformations.
#' @param validation_split Optional float between 0 and 1, fraction of data to
#'   reserve for validation.
#' @param subset One of "training", "validation", or "both" (available for TF>=2.10).
#'   Only used if validation_split is set. When `subset="both"`, the utility returns
#'   a tuple of two datasets (the training and validation datasets respectively).
#' @param interpolation String, the interpolation method used when resizing
#'   images. Defaults to bilinear. Supports bilinear, nearest, bicubic, area,
#'   lanczos3, lanczos5, gaussian, mitchellcubic.
#' @param follow_links Whether to visits subdirectories pointed to by symlinks.
#'   Defaults to FALSE.
#' @param crop_to_aspect_ratio If `TRUE`, resize the images without aspect ratio
#'   distortion. When the original aspect ratio differs from the target aspect
#'   ratio, the output image will be cropped so as to return the largest
#'   possible window in the image (of size image_size) that matches the target
#'   aspect ratio. By default (crop_to_aspect_ratio=False), aspect ratio may not
#'   be preserved.
#' @param ... Legacy arguments
#'
#'
#' @return A tf.data.Dataset object. If label_mode is `NULL`, it yields float32
#'   tensors of shape `(batch_size, image_size[1], image_size[2], num_channels)`,
#'   encoding images (see below for rules regarding `num_channels`).
#'
#'   Otherwise, it yields pairs of `(images, labels)`, where images has shape
#'   `(batch_size, image_size[1], image_size[2], num_channels)`, and labels
#'   follows the format described below.
#'
#'   Rules regarding labels format:
#'
#'   +  if label_mode is int, the labels are an int32 tensor of shape
#'   `(batch_size)`.
#'
#'   +  if label_mode is binary, the labels are a float32 tensor of 1s and 0s of
#'   shape `(batch_size, 1)`.
#'
#'   +  if label_mode is categorial, the labels are a float32 tensor of shape
#'   `(batch_size, num_classes)`, representing a one-hot encoding of the class
#'   index.
#'
#'   Rules regarding number of channels in the yielded images:
#'
#'   +  if color_mode is grayscale, there's 1 channel in the image tensors.
#'
#'   +   if color_mode is rgb, there are 3 channel in the image tensors.
#'
#'   +  if color_mode is rgba, there are 4 channel in the image tensors.
#'
#' @seealso <https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory>
#' @export
image_dataset_from_directory <- function(
  directory,
  labels="inferred",
  label_mode="int",
  class_names=NULL,
  color_mode="rgb",
  batch_size=32,
  image_size=c(256, 256),
  shuffle=TRUE,
  seed=NULL,
  validation_split=NULL,
  subset=NULL,
  interpolation="bilinear",
  follow_links=FALSE,
  crop_to_aspect_ratio = FALSE,
  ...
) {

  args <- capture_args(match.call(), list(
    directory = function(d) normalizePath(d, mustWork = FALSE),
    batch_size = as.integer,
    image_size = as_integer_tuple,
    seed = as_nullable_integer,
    labels = function(l) if(is.character(l)) l else as.integer(l)
  ))

  if (identical(subset, "both") && tensorflow::tf_version() < "2.10")
    stop('subset="both" is only available for TF>=2.10')

  out <- do.call(keras$preprocessing$image_dataset_from_directory, args)
  class(out) <- unique(c("tf_dataset", class(out)))
  out
}


#' Generate a `tf.data.Dataset` from text files in a directory
#'
#' @details
#' If your directory structure is:
#'
#' ```
#' main_directory/
#' ...class_a/
#' ......a_text_1.txt
#' ......a_text_2.txt
#' ...class_b/
#' ......b_text_1.txt
#' ......b_text_2.txt
#' ```
#'
#' Then calling `text_dataset_from_directory(main_directory, labels = 'inferred')`
#' will return a `tf.data.Dataset` that yields batches of texts from
#' the subdirectories `class_a` and `class_b`, together with labels
#' 0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).
#'
#' Only `.txt` files are supported at this time.
#'
#' @param directory Directory where the data is located.
#' If `labels` is "inferred", it should contain
#' subdirectories, each containing text files for a class.
#' Otherwise, the directory structure is ignored.
#'
#' @param labels Either "inferred"
#' (labels are generated from the directory structure),
#' NULL (no labels),
#' or a list of integer labels of the same size as the number of
#' text files found in the directory. Labels should be sorted according
#' to the alphanumeric order of the text file paths
#' (obtained via `os.walk(directory)` in Python).
#'
#' @param label_mode - `'int'`: means that the labels are encoded as integers
#'     (e.g. for `sparse_categorical_crossentropy` loss).
#' - `'categorical'` means that the labels are
#'     encoded as a categorical vector
#'     (e.g. for `categorical_crossentropy` loss).
#' - `'binary'` means that the labels (there can be only 2)
#'     are encoded as `float32` scalars with values 0 or 1
#'     (e.g. for `binary_crossentropy`).
#' - `NULL` (no labels).
#'
#' @param class_names Only valid if `labels` is `"inferred"`. This is the explicit
#' list of class names (must match names of subdirectories). Used
#' to control the order of the classes
#' (otherwise alphanumerical order is used).
#'
#' @param batch_size Size of the batches of data. Default: `32`.
#'
#' @param max_length Maximum size of a text string. Texts longer than this will
#' be truncated to `max_length`.
#'
#' @param shuffle Whether to shuffle the data. Default: `TRUE`.
#' If set to `FALSE`, sorts the data in alphanumeric order.
#'
#' @param seed Optional random seed for shuffling and transformations.
#'
#' @param validation_split Optional float between 0 and 1,
#' fraction of data to reserve for validation.
#'
#' @param subset One of "training" or "validation".
#' Only used if `validation_split` is set.
#'
#' @param follow_links Whether to visits subdirectories pointed to by symlinks.
#' Defaults to `FALSE`.
#'
#' @param ... For future compatibility (unused presently).
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/utils/text_dataset_from_directory>
#'
#' @export
text_dataset_from_directory <-
function(directory,
         labels = "inferred",
         label_mode = "int",
         class_names = NULL,
         batch_size = 32L,
         max_length = NULL,
         shuffle = TRUE,
         seed = NULL,
         validation_split = NULL,
         subset = NULL,
         follow_links = FALSE,
         ...
)
{
  args <- capture_args(match.call(),
                       list(batch_size = as.integer,
                            max_length = as_nullable_integer,
                            seed = as_nullable_integer))
  do.call(keras$preprocessing$text_dataset_from_directory, args)
}

#' Creates a dataset of sliding windows over a timeseries provided as array
#'
#' @details
#' This function takes in a sequence of data-points gathered at
#' equal intervals, along with time series parameters such as
#' length of the sequences/windows, spacing between two sequence/windows, etc.,
#' to produce batches of timeseries inputs and targets.
#'
#' @section Example 1:
#'
#' Consider indices `0:99`. With `sequence_length=10`, `sampling_rate=2`,
#' `sequence_stride=3`, `shuffle=FALSE`, the dataset will yield batches of
#' sequences composed of the following indices:
#'
#' ```
#' First sequence:  0  2  4  6  8 10 12 14 16 18
#' Second sequence: 3  5  7  9 11 13 15 17 19 21
#' Third sequence:  6  8 10 12 14 16 18 20 22 24
#' ...
#' Last sequence:   78 80 82 84 86 88 90 92 94 96
#' ```
#'
#' In this case the last 3 data points are discarded since no full sequence
#' can be generated to include them (the next sequence would have started
#' at index 81, and thus its last step would have gone over 99).
#'
#' @section Example 2: Temporal regression.
#'
#' Consider an array `data` of scalar values, of shape `(steps)`.
#' To generate a dataset that uses the past 10
#' timesteps to predict the next timestep, you would use:
#'
#' ``` R
#' steps <- 100
#' # data is integer seq with some noise
#' data <- array(1:steps + abs(rnorm(steps, sd = .25)))
#' inputs_data <- head(data, -10) # drop last 10
#' targets <- tail(data, -10)    # drop first 10
#' dataset <- timeseries_dataset_from_array(
#'   inputs_data, targets, sequence_length=10)
#' library(tfdatasets)
#' dataset_iterator <- as_iterator(dataset)
#' repeat {
#'   batch <- iter_next(dataset_iterator)
#'   if(is.null(batch)) break
#'   c(input, target) %<-% batch
#'   stopifnot(exprs = {
#'     # First sequence: steps [1-10]
#'     # Corresponding target: step 11
#'     all.equal(as.array(input[1, ]), data[1:10])
#'     all.equal(as.array(target[1]), data[11])
#'
#'     all.equal(as.array(input[2, ]), data[2:11])
#'     all.equal(as.array(target[2]), data[12])
#'
#'     all.equal(as.array(input[3, ]), data[3:12])
#'     all.equal(as.array(target[3]), data[13])
#'   })
#' }
#' ```
#'
#' @section Example 3: Temporal regression for many-to-many architectures.
#'
#' Consider two arrays of scalar values `X` and `Y`,
#' both of shape `(100)`. The resulting dataset should consist of samples with
#' 20 timestamps each. The samples should not overlap.
#' To generate a dataset that uses the current timestamp
#' to predict the corresponding target timestep, you would use:
#'
#' ``` R
#' X <- seq(100)
#' Y <- X*2
#'
#' sample_length <- 20
#' input_dataset <- timeseries_dataset_from_array(
#'   X, NULL, sequence_length=sample_length, sequence_stride=sample_length)
#' target_dataset <- timeseries_dataset_from_array(
#'   Y, NULL, sequence_length=sample_length, sequence_stride=sample_length)
#'
#' library(tfdatasets)
#' dataset_iterator <-
#'   zip_datasets(input_dataset, target_dataset) %>%
#'   as_array_iterator()
#' while(!is.null(batch <- iter_next(dataset_iterator))) {
#'   c(inputs, targets) %<-% batch
#'   stopifnot(
#'     all.equal(inputs[1,], X[1:sample_length]),
#'     all.equal(targets[1,], Y[1:sample_length]),
#'     # second sample equals output timestamps 20-40
#'     all.equal(inputs[2,], X[(1:sample_length) + sample_length]),
#'     all.equal(targets[2,], Y[(1:sample_length) + sample_length])
#'   )
#' }
#' ```
#'
#' @param data array or eager tensor
#' containing consecutive data points (timesteps).
#' The first axis is expected to be the time dimension.
#'
#' @param targets Targets corresponding to timesteps in `data`.
#' `targets[i]` should be the target
#' corresponding to the window that starts at index `i`
#' (see example 2 below).
#' Pass NULL if you don't have target data (in this case the dataset will
#' only yield the input data).
#'
#' @param sequence_length Length of the output sequences (in number of timesteps).
#'
#' @param sequence_stride Period between successive output sequences.
#' For stride `s`, output samples would
#' start at index `data[i]`, `data[i + s]`, `data[i + (2 * s)]`, etc.
#'
#' @param sampling_rate Period between successive individual timesteps
#' within sequences. For rate `r`, timesteps
#' `data[i], data[i + r], ... data[i + sequence_length]`
#' are used for create a sample sequence.
#'
#' @param batch_size Number of timeseries samples in each batch
#' (except maybe the last one).
#'
#' @param shuffle Whether to shuffle output samples,
#' or instead draw them in chronological order.
#'
#' @param seed Optional int; random seed for shuffling.
#'
#' @param start_index,end_index Optional int (1 based); data points earlier
#' than `start_index` or later then `end_index` will not be used
#' in the output sequences. This is useful to reserve part of the
#' data for test or validation.
#'
#'
#'
#' @param ... For backwards and forwards compatibility, ignored presently.
#'
#' @seealso
#'   +  <https://www.tensorflow.org/api_docs/python/tf/keras/utils/timeseries_dataset_from_array>
#'
#' @returns A `tf.data.Dataset` instance. If `targets` was passed, the
#'   dataset yields batches of two items: `(batch_of_sequences,
#'   batch_of_targets)`. If not, the dataset yields only
#'   `batch_of_sequences`.
#'
#'
#' @section Example:
#'
#' ````
#' int_sequence <- seq(20)
#'
#' dummy_dataset <- timeseries_dataset_from_array(
#'   data = head(int_sequence, -3), # drop last 3
#'   targets = tail(int_sequence, -3), # drop first 3
#'   sequence_length = 3,
#'   start_index = 3,
#'   end_index = 9,
#'   batch_size = 2
#' )
#'
#' library(tfdatasets)
#' dummy_dataset_iterator <- as_array_iterator(dummy_dataset)
#'
#' repeat {
#'   batch <- iter_next(dummy_dataset_iterator)
#'   if (is.null(batch)) # iterator exhausted
#'     break
#'   c(inputs, targets) %<-% batch
#'   for (r in 1:nrow(inputs))
#'     cat(sprintf("input: [ %s ]  target: %s\n",
#'                 paste(inputs[r,], collapse = " "), targets[r]))
#'   cat("---------------------------\n") # demark batchs
#' }
#' ````
#' Will give output like:
#' ````
#' input: [ 3 4 5 ]  target: 6
#' input: [ 4 5 6 ]  target: 7
#' ---------------------------
#' input: [ 5 6 7 ]  target: 8
#' input: [ 6 7 8 ]  target: 9
#' ---------------------------
#' input: [ 7 8 9 ]  target: 10
#' ````
#'
#'
#' @export
timeseries_dataset_from_array <-
function(data, targets, sequence_length, sequence_stride = 1L,
         sampling_rate = 1L, batch_size = 128L, shuffle = FALSE, ...,
         seed = NULL, start_index = NULL, end_index = NULL)
{
  # start_index and end_index are 0-based
  require_tf_version("2.6", "timeseries_dataset_from_array")

  args <- capture_args(match.call(), list(
    data = keras_array,
    targets = keras_array,
    sequence_length = as.integer,
    sequence_stride = as.integer,
    sampling_rate = as.integer,
    batch_size = as.integer,
    seed = as_nullable_integer,
    start_index = as_slice_start,
    end_index = as_slice_end
  ))
  do.call(keras$preprocessing$timeseries_dataset_from_array, args)
}


as_slice_start <- function(x) {
  if (is.null(x))
    return(x)
  x <- as.integer(x)
  if (x >= 1)
    x <- x - 1L
  x
}


as_slice_end <- function(x) {
  if(is.null(x))
    return(x)
  x <- as.integer(x)
  if(x == -1L)
    return(NULL)
  if(x < 0)
    x <- x + 1L
  x
}
