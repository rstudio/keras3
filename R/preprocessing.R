
#' Pads each sequence to the same length (length of the longest sequence).
#' 
#' @details 
#' If maxlen is provided, any sequence longer than maxlen is truncated to maxlen.
#' Truncation happens off either the beginning (default) or
#' the end of the sequence. Supports post-padding and pre-padding (default).
#' 
#' @param sequences List of lists where each element is a sequence
#' @param maxlen int, maximum length
#' @param dtype type to cast the resulting sequence.
#' @param padding 'pre' or 'post', pad either before or after each sequence.
#' @param truncating 'pre' or 'post', remove values from sequences larger than maxlen either in the beginning or in the end of the sequence
#' @param value float, value to pad the sequences to the desired value.
#' 
#' @return Array with dimensions (number_of_sequences, maxlen)
#'
#' @family text preprocessing
#'
#' @export
pad_sequences <- function(sequences, maxlen = NULL, dtype = "int32", padding = "pre", 
                          truncating = "pre", value = 0.0) {
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
#' Takes a sequence (list of indexes of words), returns list of `couples` (word_index,
#' other_word index) and `labels` (1s or 0s), where label = 1 if 'other_word'
#' belongs to the context of 'word', and label=0 if 'other_word' is randomly
#' sampled
#' 
#' @param sequence a word sequence (sentence), encoded as a list of word indices
#'   (integers). If using a `sampling_table`, word indices are expected to match
#'   the rank of the words in a reference dataset (e.g. 10 would encode the
#'   10-th most frequently occuring token). Note that index 0 is expected to be
#'   a non-word and will be skipped.
#' @param vocabulary_size int. maximum possible word index + 1
#' @param window_size int. actually half-window. The window of a word wi will be
#'   `[i-window_size, i+window_size+1]`
#' @param negative_samples float >= 0. 0 for no negative (=random) samples. 1
#'   for same number as positive samples. etc.
#' @param shuffle whether to shuffle the word couples before returning them.
#' @param categorical bool. if FALSE, labels will be integers (eg. `[0, 1, 1 .. ]`), 
#'   if TRUE labels will be categorical eg. `[[1,0],[0,1],[0,1] .. ]`
#' @param sampling_table 1D array of size `vocabulary_size` where the entry i
#'   encodes the probabibily to sample a word of rank i.
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
                      shuffle = TRUE, categorical = FALSE, sampling_table = NULL) {
  sg <- keras$preprocessing$sequence$skipgrams(
    sequence = as.integer(sequence),
    vocabulary_size = as.integer(vocabulary_size),
    window_size = as.integer(window_size),
    negative_samples = negative_samples,
    shuffle = shuffle,
    categorical = categorical,
    sampling_table = sampling_table
  )
  sg <- list(
    couples = sg[[1]],
    labels = sg[[2]]
  )
}


#' Generates a word rank-based probabilistic sampling table.
#' 
#' This generates an array where the ith element is the probability that a word
#' of rank i would be sampled, according to the sampling distribution used in
#' word2vec. The word2vec formula is: p(word) = min(1,
#' sqrt(word.frequency/sampling_factor) / (word.frequency/sampling_factor)) We
#' assume that the word frequencies follow Zipf's law (s=1) to derive a
#' numerical approximation of frequency(rank): frequency(rank) ~ 1/(rank *
#' (log(rank) + gamma) + 1/2 - 1/(12*rank)) where gamma is the Euler-Mascheroni
#' constant.
#' 
#' @param size int, number of possible words to sample.
#' @param sampling_factor the sampling factor in the word2vec formula.
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

#' Convert text to a sequence of word indices.
#' 
#' @param text Input text (string).
#' @param filters Sequence of characters to filter out.
#' @param lower Whether to convert the input to lowercase.
#' @param split Sentence split marker (string).
#' 
#' @return integer word indices.
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
#' @param char_level if True, every character will be treated as a word.
#'   
#' @family text tokenization
#'   
#' @export
text_tokenizer <- function(num_words = NULL, filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                           lower = TRUE, split = ' ', char_level = FALSE) {
  keras$preprocessing$text$Tokenizer(
    num_words = as_nullable_integer(num_words),
    filters = filters,
    lower = lower,
    split = split,
    char_level = char_level
  )
}

#' Update tokenizer internal vocabulary based on a list of texts.
#' 
#' @param tokenizer Tokenizer returned by [text_tokenizer()]
#' @param texts Vector/list of strings, or a generator of strings (for memory-efficiency)
#' 
#' @note 
#' Required before using [texts_to_sequences()] or [texts_to_matrix()].
#' 
#' @family text tokenization
#'   
#' @export
fit_on_texts <- function(tokenizer, texts) {
  tokenizer$fit_on_texts(texts)
}

#' Update tokenizer internal vocabulary based on a list of sequences.
#' 
#' @inheritParams fit_on_texts
#' 
#' @param sequences A list of sequence (a "sequence" is a list of integer word indices).
#' 
#' @note 
#' Required before using [sequences_to_matrix()] (if [fit_on_texts()] was never called).
#' 
#' @family text tokenization
#'   
#' @export
fit_on_sequences <- function(tokenizer, sequences) {
  tokenizer$fit_on_sequences
}

#' Transform each text in texts in a sequence of integers.
#'
#' Only top "num_words" most frequent words will be taken into account.
#' Only words known by the tokenizer will be taken into account.
#' 
#' @inheritParams fit_on_texts
#' 
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
  tokenizer$sequences_to_matrix(
    sequences = sequences,
    mode = mode
  )
}

