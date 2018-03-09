#' Sequence to sequence example in Keras (character-level).
#'
#' This script demonstrates how to implement a basic character-level
#' sequence-to-sequence model. We apply it to translating
#' short English sentences into short French sentences,
#' character-by-character. Note that it is fairly unusual to
#' do character-level machine translation, as word-level
#' models are more common in this domain.
#'
#' **Algorithm**
#'
#' - We start with input sequences from a domain (e.g. English sentences)
#'     and correspding target sequences from another domain
#'     (e.g. French sentences).
#' - An encoder LSTM turns input sequences to 2 state vectors
#'     (we keep the last LSTM state and discard the outputs).
#' - A decoder LSTM is trained to turn the target sequences into
#'     the same sequence but offset by one timestep in the future,
#'     a training process called "teacher forcing" in this context.
#'     Is uses as initial state the state vectors from the encoder.
#'     Effectively, the decoder learns to generate `targets[t+1...]`
#'     given `targets[...t]`, conditioned on the input sequence.
#' - In inference mode, when we want to decode unknown input sequences, we:
#'     - Encode the input sequence into state vectors
#'     - Start with a target sequence of size 1
#'         (just the start-of-sequence character)
#'     - Feed the state vectors and 1-char target sequence
#'         to the decoder to produce predictions for the next character
#'     - Sample the next character using these predictions
#'         (we simply use argmax).
#'     - Append the sampled character to the target sequence
#'     - Repeat until we generate the end-of-sequence character or we
#'         hit the character limit.
#'
#' **Data download**
#'
#' English to French sentence pairs.
#' http://www.manythings.org/anki/fra-eng.zip
#'
#' Lots of neat sentence pairs datasets can be found at:
#' http://www.manythings.org/anki/
#'
#' **References**
#'
#' - Sequence to Sequence Learning with Neural Networks
#'     https://arxiv.org/abs/1409.3215
#' - Learning Phrase Representations using
#'     RNN Encoder-Decoder for Statistical Machine Translation
#'     https://arxiv.org/abs/1406.1078

library(keras)
library(data.table)

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

## Path to the data txt file on disk.
data_path = 'fra.txt'
text <- fread(data_path, sep="\t", header=FALSE, nrows=num_samples)

## Vectorize the data.
input_texts  <- text[[1]]
target_texts <- paste0('\t',text[[2]],'\n')
input_texts  <- lapply( input_texts, function(s) strsplit(s, split="")[[1]])
target_texts <- lapply( target_texts, function(s) strsplit(s, split="")[[1]])

input_characters  <- sort(unique(unlist(input_texts)))
target_characters <- sort(unique(unlist(target_texts)))
num_encoder_tokens <- length(input_characters)
num_decoder_tokens <- length(target_characters)
max_encoder_seq_length <- max(sapply(input_texts,length))
max_decoder_seq_length <- max(sapply(target_texts,length))

cat('Number of samples:', length(input_texts),'\n')
cat('Number of unique input tokens:', num_encoder_tokens,'\n')
cat('Number of unique output tokens:', num_decoder_tokens,'\n')
cat('Max sequence length for inputs:', max_encoder_seq_length,'\n')
cat('Max sequence length for outputs:', max_decoder_seq_length,'\n')

input_token_index  <- 1:length(input_characters)
names(input_token_index) <- input_characters
target_token_index <- 1:length(target_characters)
names(target_token_index) <- target_characters
encoder_input_data <- array(
  0, dim = c(length(input_texts), max_encoder_seq_length, num_encoder_tokens))
decoder_input_data <- array(
  0, dim = c(length(input_texts), max_decoder_seq_length, num_decoder_tokens))
decoder_target_data <- array(
  0, dim = c(length(input_texts), max_decoder_seq_length, num_decoder_tokens))

for(i in 1:length(input_texts)) {
  d1 <- sapply( input_characters, function(x) { as.integer(x == input_texts[[i]]) })
  encoder_input_data[i,1:nrow(d1),] <- d1
  d2 <- sapply( target_characters, function(x) { as.integer(x == target_texts[[i]]) })
  decoder_input_data[i,1:nrow(d2),] <- d2
  d3 <- sapply( target_characters, function(x) { as.integer(x == target_texts[[i]][-1]) })
  decoder_target_data[i,1:nrow(d3),] <- d3
}

##----------------------------------------------------------------------
## Create the model
##----------------------------------------------------------------------

## Define an input sequence and process it.
encoder_inputs  <- layer_input(shape=list(NULL,num_encoder_tokens))
encoder         <- layer_lstm(units=latent_dim, return_state=TRUE)
encoder_results <- encoder_inputs %>% encoder
## We discard `encoder_outputs` and only keep the states.
encoder_states  <- encoder_results[2:3]

## Set up the decoder, using `encoder_states` as initial state.
decoder_inputs  <- layer_input(shape=list(NULL, num_decoder_tokens))
## We set up our decoder to return full output sequences,
## and to return internal states as well. We don't use the
## return states in the training model, but we will use them in inference.
decoder_lstm    <- layer_lstm(units=latent_dim, return_sequences=TRUE,
                              return_state=TRUE, stateful=FALSE)
decoder_results <- decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense   <- layer_dense(units=num_decoder_tokens, activation='softmax')
decoder_outputs <- decoder_dense(decoder_results[[1]])

## Define the model that will turn
## `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model <- keras_model( inputs = list(encoder_inputs, decoder_inputs),
                      outputs = decoder_outputs )

## Compile model
model %>% compile(optimizer='rmsprop', loss='categorical_crossentropy')

## Run model
model %>% fit( list(encoder_input_data, decoder_input_data), decoder_target_data,
               batch_size=batch_size,
               epochs=epochs,
               validation_split=0.2)

## Save model
save_model_hdf5(model,'s2s.h5')
save_model_weights_hdf5(model,'s2s-wt.h5')

##model <- load_model_hdf5('s2s.h5')
##load_model_weights_hdf5(model,'s2s-wt.h5')

##----------------------------------------------------------------------
## Next: inference mode (sampling).
##----------------------------------------------------------------------
## Here's the drill:
## 1) encode input and retrieve initial decoder state
## 2) run one step of decoder with this initial state
## and a "start of sequence" token as target.
## Output will be the next target token
## 3) Repeat with the current target token and current states

## Define sampling models
encoder_model <-  keras_model(encoder_inputs, encoder_states)
decoder_state_input_h <- layer_input(shape=latent_dim)
decoder_state_input_c <- layer_input(shape=latent_dim)
decoder_states_inputs <- c(decoder_state_input_h, decoder_state_input_c)
decoder_results <- decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states  <- decoder_results[2:3]
decoder_outputs <- decoder_dense(decoder_results[[1]])
decoder_model   <- keras_model(
  inputs  = c(decoder_inputs, decoder_states_inputs),
  outputs = c(decoder_outputs, decoder_states))

## Reverse-lookup token index to decode sequences back to
## something readable.
reverse_input_char_index  <- as.character(input_characters)
reverse_target_char_index <- as.character(target_characters)

decode_sequence <- function(input_seq) {
  ## Encode the input as state vectors.
  states_value <- predict(encoder_model, input_seq)
  
  ## Generate empty target sequence of length 1.
  target_seq <- array(0, dim=c(1, 1, num_decoder_tokens))
  ## Populate the first character of target sequence with the start character.
  target_seq[1, 1, target_token_index['\t']] <- 1.
  
  ## Sampling loop for a batch of sequences
  ## (to simplify, here we assume a batch of size 1).
  stop_condition = FALSE
  decoded_sentence = ''
  maxiter = max_decoder_seq_length
  niter = 1
  while (!stop_condition && niter < maxiter) {
    
    ## output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
    decoder_predict <- predict(decoder_model, c(list(target_seq), states_value))
    output_tokens <- decoder_predict[[1]]
    
    ## Sample a token
    sampled_token_index <- which.max(output_tokens[1, 1, ])
    sampled_char <- reverse_target_char_index[sampled_token_index]
    decoded_sentence <-  paste0(decoded_sentence, sampled_char)
    decoded_sentence
    
    ## Exit condition: either hit max length
    ## or find stop character.
    if (sampled_char == '\n' ||
        length(decoded_sentence) > max_decoder_seq_length) {
      stop_condition = TRUE
    }
    
    ## Update the target sequence (of length 1).
    ## target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[1, 1, ] <- 0
    target_seq[1, 1, sampled_token_index] <- 1.
    
    ## Update states
    h <- decoder_predict[[2]]
    c <- decoder_predict[[3]]
    states_value = list(h, c)
    niter <- niter + 1
  }    
  return(decoded_sentence)
}

for (seq_index in 1:100) {
  ## Take one sequence (part of the training test)
  ## for trying out decoding.
  input_seq = encoder_input_data[seq_index,,,drop=FALSE]
  decoded_sentence = decode_sequence(input_seq)
  target_sentence <- gsub("\t|\n","",paste(target_texts[[seq_index]],collapse=''))
  input_sentence  <- paste(input_texts[[seq_index]],collapse='')
  cat('-\n')
  cat('Input sentence  : ', input_sentence,'\n')
  cat('Target sentence : ', target_sentence,'\n')
  cat('Decoded sentence: ', decoded_sentence,'\n')
}


