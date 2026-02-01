# English-to-Spanish translation with a sequence-to-sequence Transformer

## Introduction

In this example, we’ll build a sequence-to-sequence Transformer model,
which we’ll train on an English-to-Spanish machine translation task.

You’ll learn how to:

- Vectorize text using
  [`layer_text_vectorization()`](https://keras3.posit.co/reference/layer_text_vectorization.md).
- Implement a `layer_transformer_encoder()`, a
  `layer_transformer_decoder()`, and a `layer_positional_embedding()`.
- Prepare data for training a sequence-to-sequence model.
- Use the trained model to generate translations of never-seen-before
  input sentences (sequence-to-sequence inference).

The code featured here is adapted from the book [Deep Learning with R,
Second
Edition](https://www.manning.com/books/deep-learning-with-r-second-edition)
(chapter 11: Deep learning for text). The present example is fairly
barebones, so for detailed explanations of how each building block
works, as well as the theory behind Transformers, I recommend reading
the book.

## Setup

``` r
library(glue)

library(tensorflow, exclude = c("set_random_seed", "shape"))
library(keras3)
```

## Downloading the data

We’ll be working with an English-to-Spanish translation dataset provided
by [Anki](https://www.manythings.org/anki/). Let’s download it:

``` r
zip_path <-
  "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip" |>
  get_file(origin = _, extract = TRUE)

text_path <- fs::path(zip_path, "spa-eng/spa.txt")
```

## Parsing the data

Each line contains an English sentence and its corresponding Spanish
sentence. The English sentence is the *source sequence* and Spanish one
is the *target sequence*. We prepend the token `"[start]"` and we append
the token `"[end]"` to the Spanish sentence.

``` r
text_file <- "spa-eng/spa.txt"
text_pairs <- text_file %>%
  readr::read_tsv(col_names = c("english", "spanish"),
                  col_types = c("cc")) %>%
  within(spanish %<>% paste("[start]", ., "[end]"))
```

Here’s what our sentence pairs look like:

``` r
df <- text_pairs[sample(nrow(text_pairs), 5), ]
glue::glue_data(df, r"(
  english: {english}
  spanish: {spanish}
)") |> cat(sep = "\n\n")
```

    ## english: I'm staying in Italy.
    ## spanish: [start] Me estoy quedando en Italia. [end]
    ##
    ## english: What's so strange about that?
    ## spanish: [start] ¿Qué es tan extraño acerca de eso? [end]
    ##
    ## english: All of the buses are full.
    ## spanish: [start] Todos los bondis están llenos. [end]
    ##
    ## english: Is this where your mother works?
    ## spanish: [start] ¿Es aquí donde trabaja tu madre? [end]
    ##
    ## english: Take precautions.
    ## spanish: [start] Ten cuidado. [end]

Now, let’s split the sentence pairs into a training set, a validation
set, and a test set.

``` python
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")
```

``` r
num_test_samples <- num_val_samples <-
  round(0.15 * nrow(text_pairs))
num_train_samples <- nrow(text_pairs) - num_val_samples - num_test_samples

pair_group <- sample(c(
  rep("train", num_train_samples),
  rep("test", num_test_samples),
  rep("val", num_val_samples)
))

train_pairs <- text_pairs[pair_group == "train", ]
test_pairs <- text_pairs[pair_group == "test", ]
val_pairs <- text_pairs[pair_group == "val", ]
glue(r"(
  {nrow(text_pairs)} total pairs
  {nrow(train_pairs)} training pairs
  {nrow(val_pairs)} validation pairs
  {nrow(test_pairs)} test pairs
)", .transformer = function(text, envir) {
  val <- eval(str2lang(text), envir)
  prettyNum(val, big.mark = ",")
})
```

    ## 118,493 total pairs
    ## 82,945 training pairs
    ## 17,774 validation pairs
    ## 17,774 test pairs

## Vectorizing the text data

We’ll use two instances of
[`layer_text_vectorization()`](https://keras3.posit.co/reference/layer_text_vectorization.md)
to vectorize the text data (one for English and one for Spanish), that
is to say, to turn the original strings into integer sequences where
each integer represents the index of a word in a vocabulary.

The English layer will use the default string standardization (strip
punctuation characters) and splitting scheme (split on whitespace),
while the Spanish layer will use a custom standardization, where we add
the character `"¿"` to the set of punctuation characters to be stripped.

Note: in a production-grade machine translation model, I would not
recommend stripping the punctuation characters in either language.
Instead, I would recommend turning each punctuation character into its
own token, which you could achieve by providing a custom `split`
function to
[`layer_text_vectorization()`](https://keras3.posit.co/reference/layer_text_vectorization.md).

``` r
punctuation_regex <- "[¡¿]|[^[:^punct:][\\]]"
# the regex explained: Match ¡, or ¿, or any punctuation character except ]
#
# [:^punct:]: is a negated POSIX character class.
# [:punct:] matches any punctuation character, so [:^punct:] matches any
# character that is not a punctuation character.
# [^...] negates the whole character class
# So [^[:^punct:]] would matche any character that is a punctuation character.
# Putting this all together, [^[:^punct:][\\]] matches any
# punctuation character except the ] character.

custom_standardization <- function(input_string) {
  input_string %>%
    tf$strings$lower() %>%
    tf$strings$regex_replace(punctuation_regex, "")
}

input_string <- as_tensor("[start] ¡corre! [end]")
custom_standardization(input_string)
```

    ## tf.Tensor(b'[start] corre [end]', shape=(), dtype=string)

``` r
vocab_size <- 15000
sequence_length <- 20

# rename to eng_vectorization
eng_vectorization <- layer_text_vectorization(
  max_tokens = vocab_size,
  output_mode = "int",
  output_sequence_length = sequence_length
)

spa_vectorization <- layer_text_vectorization(
  max_tokens = vocab_size,
  output_mode = "int",
  output_sequence_length = sequence_length + 1,
  standardize = custom_standardization
)

adapt(eng_vectorization, train_pairs$english)
adapt(spa_vectorization, train_pairs$spanish)
```

Next, we’ll format our datasets.

At each training step, the model will seek to predict target words N+1
(and beyond) using the source sentence and the target words from 1 to N.

As such, the training dataset will yield a tuple `(inputs, targets)`,
where:

- `inputs` is a dictionary (named list) with the keys (names)
  `encoder_inputs` and `decoder_inputs`. `encoder_inputs` is the
  vectorized source sentence and `decoder_inputs` is the target sentence
  “so far”, that is to say, the words 0 to N used to predict word N+1
  (and beyond) in the target sentence.
- `target` is the target sentence offset by one step: it provides the
  next words in the target sentence – what the model will try to
  predict.

``` r
format_pair <- function(pair) {
  eng <- pair$english |> eng_vectorization()
  spa <- pair$spanish |> spa_vectorization()

  spa_feature <- spa@r[NA:-2]                                                   # <1>
  spa_target <- spa@r[2:NA]                                                     # <2>

  features <- list(encoder_inputs = eng, decoder_inputs = spa_feature)
  labels <- spa_target
  sample_weight <- labels != 0

  tuple(features, labels, sample_weight)
}

batch_size <- 64

library(tfdatasets, exclude = "shape")
make_dataset <- function(pairs) {
  tensor_slices_dataset(pairs) |>
    dataset_map(format_pair, num_parallel_calls = 4) |>
    dataset_cache() |>
    dataset_shuffle(2048) |>
    dataset_batch(batch_size) |>
    dataset_prefetch(16)
}

train_ds <- make_dataset(train_pairs)
val_ds <- make_dataset(val_pairs)
```

Let’s take a quick look at the sequence shapes (we have batches of 64
pairs, and all sequences are 20 steps long):

``` r
c(inputs, targets, weights) %<-% iter_next(as_iterator(train_ds))
str(inputs)
```

    ## List of 2
    ##  $ encoder_inputs:<tf.Tensor: shape=(64, 20), dtype=int64, numpy=…>
    ##  $ decoder_inputs:<tf.Tensor: shape=(64, 20), dtype=int64, numpy=…>

``` r
str(targets)
```

    ## <tf.Tensor: shape=(64, 20), dtype=int64, numpy=…>

## Building the model

Our sequence-to-sequence Transformer consists of a `TransformerEncoder`
and a `TransformerDecoder` chained together. To make the model aware of
word order, we also use a `PositionalEmbedding` layer.

The source sequence will be pass to the `TransformerEncoder`, which will
produce a new representation of it. This new representation will then be
passed to the `TransformerDecoder`, together with the target sequence so
far (target words 1 to N). The `TransformerDecoder` will then seek to
predict the next words in the target sequence (N+1 and beyond).

A key detail that makes this possible is causal masking (see method
`get_causal_attention_mask()` on the `TransformerDecoder`). The
`TransformerDecoder` sees the entire sequences at once, and thus we must
make sure that it only uses information from target tokens 0 to N when
predicting token N+1 (otherwise, it could use information from the
future, which would result in a model that cannot be used at inference
time).

``` r
layer_transformer_encoder <- Layer(
  classname = "TransformerEncoder",
  initialize = function(embed_dim, dense_dim, num_heads, ...) {
    super$initialize(...)
    self$embed_dim <- embed_dim
    self$dense_dim <- dense_dim
    self$num_heads <- num_heads
    self$attention <-
      layer_multi_head_attention(num_heads = num_heads,
                                 key_dim = embed_dim)

    self$dense_proj <- keras_model_sequential() %>%
      layer_dense(dense_dim, activation = "relu") %>%
      layer_dense(embed_dim)

    self$layernorm_1 <- layer_layer_normalization()
    self$layernorm_2 <- layer_layer_normalization()
    self$supports_masking <- TRUE
  },

  call = function(inputs, mask = NULL) {
    if (!is.null(mask))
      mask <- mask[, NULL, ] |> op_cast("int32")

    inputs %>%
      { self$attention(., ., attention_mask = mask) + . } %>%
      self$layernorm_1() %>%
      { self$dense_proj(.) + . } %>%
      self$layernorm_2()
  },

  get_config = function() {
    config <- super$get_config()
    for(name in c("embed_dim", "num_heads", "dense_dim"))
      config[[name]] <- self[[name]]
    config
  }
)

layer_transformer_decoder <- Layer(
  classname = "TransformerDecoder",

  initialize = function(embed_dim, latent_dim, num_heads, ...) {
    super$initialize(...)
    self$embed_dim <- embed_dim
    self$latent_dim <- latent_dim
    self$num_heads <- num_heads
    self$attention_1 <- layer_multi_head_attention(num_heads = num_heads,
                                                   key_dim = embed_dim)
    self$attention_2 <- layer_multi_head_attention(num_heads = num_heads,
                                                   key_dim = embed_dim)
    self$dense_proj <- keras_model_sequential() %>%
      layer_dense(latent_dim, activation = "relu") %>%
      layer_dense(embed_dim)

    self$layernorm_1 <- layer_layer_normalization()
    self$layernorm_2 <- layer_layer_normalization()
    self$layernorm_3 <- layer_layer_normalization()
    self$supports_masking <- TRUE
  },

  get_config = function() {
    config <- super$get_config()
    for (name in c("embed_dim", "num_heads", "latent_dim"))
      config[[name]] <- self[[name]]
    config
  },


  get_causal_attention_mask = function(inputs) {
    .[batch_size, sequence_length, encoding_length] <- op_shape(inputs)

    x <- op_arange(0L, sequence_length, include_end = FALSE)
    i <- x[, newaxis]
    j <- x[newaxis, ]
    mask <- op_cast(i >= j, "int32")

    op_tile(
      mask[newaxis, , ],
      shape(batch_size, 1L, 1L)
    )
  },

  call = function(inputs, mask = NULL) {
    c(inputs_seq, encoder_outputs) %<-% inputs
    causal_mask <- self$get_causal_attention_mask(inputs_seq)

    if (is.null(mask)) {
      inputs_padding_mask <- NULL
      encoder_outputs_padding_mask <- NULL
    } else {
      c(inputs_padding_mask, encoder_outputs_padding_mask) %<-% mask
    }

    attention_output_1 <- self$attention_1(
      query = inputs_seq,
      value = inputs_seq,
      key = inputs_seq,
      attention_mask = causal_mask,
      query_mask = inputs_padding_mask
    )
    out_1 <- self$layernorm_1(inputs_seq + attention_output_1)

    attention_output_2 <- self$attention_2(
      query = out_1,
      value = encoder_outputs,
      key = encoder_outputs,
      query_mask = inputs_padding_mask,
      key_mask = encoder_outputs_padding_mask
    )
    out_2 <- self$layernorm_2(out_1 + attention_output_2)

    self$layernorm_3(out_2 + self$dense_proj(out_2))
  }
)

layer_positional_embedding <- Layer(
  classname = "PositionalEmbedding",

  initialize = function(sequence_length, vocab_size, embed_dim, ...) {
    super$initialize(...)
    self$token_embeddings <- layer_embedding(
      input_dim = vocab_size, output_dim = embed_dim
    )
    self$position_embeddings <- layer_embedding(
      input_dim = sequence_length, output_dim = embed_dim
    )
    self$sequence_length <- sequence_length
    self$vocab_size <- vocab_size
    self$embed_dim <- embed_dim
  },

  call = function(inputs) {
    .[., len] <- op_shape(inputs) # (batch_size, seq_len)
    positions <- op_arange(0, len, dtype = "int32", include_end = FALSE)
    embedded_tokens <- self$token_embeddings(inputs)
    embedded_positions <- self$position_embeddings(positions)
    embedded_tokens + embedded_positions
  },

  compute_mask = function(inputs, mask = NULL) {
    inputs != 0L
  },

  get_config = function() {
    config <- super$get_config()
    for(name in c("sequence_length", "vocab_size", "embed_dim"))
      config[[name]] <- self[[name]]
    config
  }
)
```

Next, we assemble the end-to-end model.

``` r
embed_dim <- 256
latent_dim <- 2048
num_heads <- 8

encoder_inputs <- layer_input(shape(NA), dtype = "int64",
                              name = "encoder_inputs")
encoder_outputs <- encoder_inputs %>%
  layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%
  layer_transformer_encoder(embed_dim, latent_dim, num_heads)

encoder <- keras_model(encoder_inputs, encoder_outputs)

decoder_inputs <-  layer_input(shape(NA), dtype = "int64",
                               name = "decoder_inputs")
encoded_seq_inputs <- layer_input(shape(NA, embed_dim),
                                  name = "decoder_state_inputs")

transformer_decoder <- layer_transformer_decoder(NULL,
  embed_dim, latent_dim, num_heads)

decoder_outputs <- decoder_inputs %>%
  layer_positional_embedding(sequence_length, vocab_size, embed_dim) %>%
  { transformer_decoder(list(., encoded_seq_inputs)) } %>%
  layer_dropout(0.5) %>%
  layer_dense(vocab_size, activation="softmax")

decoder <- keras_model(inputs = list(decoder_inputs, encoded_seq_inputs),
                       outputs = decoder_outputs)

decoder_outputs <- decoder(list(decoder_inputs, encoder_outputs))

transformer <- keras_model(
  inputs = list(encoder_inputs = encoder_inputs, decoder_inputs = decoder_inputs),
  outputs = decoder_outputs,
  name = "transformer"
)
```

## Training our model

We’ll use accuracy as a quick way to monitor training progress on the
validation data. Note that machine translation typically uses BLEU
scores as well as other metrics, rather than accuracy.

Here we only train for 1 epoch, but to get the model to actually
converge you should train for at least 30 epochs.

``` r
epochs <- 1  # This should be at least 30 for convergence

transformer
```

    ## Model: "transformer"
    ## ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
    ## ┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
    ## ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
    ## │ encoder_inputs      │ (None, None)      │          0 │ -                 │
    ## │ (InputLayer)        │                   │            │                   │
    ## ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    ## │ positional_embeddi… │ (None, None, 256) │  3,845,120 │ encoder_inputs[0… │
    ## │ (PositionalEmbeddi… │                   │            │                   │
    ## ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    ## │ not_equal           │ (None, None)      │          0 │ encoder_inputs[0… │
    ## │ (NotEqual)          │                   │            │                   │
    ## ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    ## │ decoder_inputs      │ (None, None)      │          0 │ -                 │
    ## │ (InputLayer)        │                   │            │                   │
    ## ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    ## │ transformer_encoder │ (None, None, 256) │  3,155,456 │ positional_embed… │
    ## │ (TransformerEncode… │                   │            │ not_equal[0][0]   │
    ## ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
    ## │ functional_3        │ (None, None,      │ 12,959,640 │ decoder_inputs[0… │
    ## │ (Functional)        │ 15000)            │            │ transformer_enco… │
    ## │                     │                   │            │ not_equal[0][0]   │
    ## └─────────────────────┴───────────────────┴────────────┴───────────────────┘
    ##  Total params: 19,960,216 (76.14 MB)
    ##  Trainable params: 19,960,216 (76.14 MB)
    ##  Non-trainable params: 0 (0.00 B)

``` r
transformer |> compile(
  "rmsprop",
  loss = loss_sparse_categorical_crossentropy(ignore_class = 0),
  metrics = "accuracy"
)

transformer |> fit(train_ds, epochs = epochs,
                   validation_data = val_ds)
```

    ## 1297/1297 - 57s - 44ms/step - accuracy: 0.3957 - loss: 3.8895 - val_accuracy: 0.3890 - val_loss: 3.3673

## Decoding test sentences

Finally, let’s demonstrate how to translate brand new English sentences.
We simply feed into the model the vectorized English sentence as well as
the target token `"[start]"`, then we repeatedly generated the next
token, until we hit the token `"[end]"`.

``` r
spa_vocab <- spa_vectorization |> get_vocabulary()
max_decoded_sentence_length <- 20
tf_decode_sequence <- tf_function(function(input_sentence) {
  withr::local_options(tensorflow.extract.style = "python")

  tokenized_input_sentence <- input_sentence %>%
    as_tensor(shape = c(1, 1)) %>%
    eng_vectorization()
  spa_vocab <- as_tensor(spa_vocab)
  decoded_sentence <- as_tensor("[start]", shape = c(1, 1))

  for (i in tf$range(as.integer(max_decoded_sentence_length))) {

    tokenized_target_sentence <-
      spa_vectorization(decoded_sentence)[,NA:-1]

    next_token_predictions <-
      transformer(list(
        encoder_inputs = tokenized_input_sentence,
        decoder_inputs = tokenized_target_sentence
      ))

    sampled_token_index <- tf$argmax(next_token_predictions[0, i, ])
    sampled_token <- spa_vocab[sampled_token_index]
    decoded_sentence <-
      tf$strings$join(c(decoded_sentence, sampled_token),
                      separator = " ")

    if (sampled_token == "[end]")
      break
  }

  decoded_sentence

})

for (i in seq(20)) {

    c(input_sentence, correct_translation) %<-%
      test_pairs[sample.int(nrow(test_pairs), 1), ]
    cat("-\n")
    cat("English:", input_sentence, "\n")
    cat("Correct Translation:", tolower(correct_translation), "\n")
    cat("  Model Translation:", input_sentence %>% as_tensor() %>%
          tf_decode_sequence() %>% as.character(), "\n")
}
```

After 30 epochs, we get results such as:

    English: I'm sure everything will be fine.
    Correct Translation: [start] estoy segura de que todo irá bien. [end]
      Model Translation: [start] estoy seguro de que todo va bien [end]
