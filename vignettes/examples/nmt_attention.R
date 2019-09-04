#' This is the companion code to the post 
#' "Attention-based Neural Machine Translation with Keras"
#' on the TensorFlow for R blog.
#' 
#' https://blogs.rstudio.com/tensorflow/posts/2018-07-30-attention-layer/

library(tensorflow)
library(keras)
library(tfdatasets)

library(purrr)
library(stringr)
library(reshape2)
library(viridis)
library(ggplot2)
library(tibble)


# Preprocessing -----------------------------------------------------------

# Assumes you've downloaded and unzipped one of the bilingual datasets offered at
# http://www.manythings.org/anki/ and put it into a directory "data"
# This example translates English to Dutch.

filepath <- file.path("data", "nld.txt")

lines <- readLines(filepath, n = 10000)
sentences <- str_split(lines, "\t")

space_before_punct <- function(sentence) {
  str_replace_all(sentence, "([?.!])", " \\1")
}

replace_special_chars <- function(sentence) {
  str_replace_all(sentence, "[^a-zA-Z?.!,¿]+", " ")
}

add_tokens <- function(sentence) {
  paste0("<start> ", sentence, " <stop>")
}
add_tokens <- Vectorize(add_tokens, USE.NAMES = FALSE)

preprocess_sentence <- compose(add_tokens,
                               str_squish,
                               replace_special_chars,
                               space_before_punct)

word_pairs <- map(sentences, preprocess_sentence)

create_index <- function(sentences) {
  unique_words <- sentences %>% unlist() %>% paste(collapse = " ") %>%
    str_split(pattern = " ") %>% .[[1]] %>% unique() %>% sort()
  index <- data.frame(
    word = unique_words,
    index = 1:length(unique_words),
    stringsAsFactors = FALSE
  ) %>%
    add_row(word = "<pad>",
            index = 0,
            .before = 1)
  index
}

word2index <- function(word, index_df) {
  index_df[index_df$word == word, "index"]
}
index2word <- function(index, index_df) {
  index_df[index_df$index == index, "word"]
}

src_index <- create_index(map(word_pairs, ~ .[[1]]))
target_index <- create_index(map(word_pairs, ~ .[[2]]))
sentence2digits <- function(sentence, index_df) {
  map((sentence %>% str_split(pattern = " "))[[1]], function(word)
    word2index(word, index_df))
}

sentlist2diglist <- function(sentence_list, index_df) {
  map(sentence_list, function(sentence)
    sentence2digits(sentence, index_df))
}

src_diglist <-
  sentlist2diglist(map(word_pairs, ~ .[[1]]), src_index)
src_maxlen <- map(src_diglist, length) %>% unlist() %>% max()
src_matrix <-
  pad_sequences(src_diglist, maxlen = src_maxlen,  padding = "post")

target_diglist <-
  sentlist2diglist(map(word_pairs, ~ .[[2]]), target_index)
target_maxlen <- map(target_diglist, length) %>% unlist() %>% max()
target_matrix <-
  pad_sequences(target_diglist, maxlen = target_maxlen, padding = "post")



# Train-test-split --------------------------------------------------------

train_indices <-
  sample(nrow(src_matrix), size = nrow(src_matrix) * 0.8)

validation_indices <- setdiff(1:nrow(src_matrix), train_indices)

x_train <- src_matrix[train_indices,]
y_train <- target_matrix[train_indices,]

x_valid <- src_matrix[validation_indices,]
y_valid <- target_matrix[validation_indices,]

buffer_size <- nrow(x_train)

# just for convenience, so we may get a glimpse at translation performance 
# during training
train_sentences <- sentences[train_indices]
validation_sentences <- sentences[validation_indices]
validation_sample <- sample(validation_sentences, 5)



# Hyperparameters / variables ---------------------------------------------

batch_size <- 32
embedding_dim <- 64
gru_units <- 256

src_vocab_size <- nrow(src_index)
target_vocab_size <- nrow(target_index)


# Create datasets ---------------------------------------------------------

train_dataset <-
  tensor_slices_dataset(keras_array(list(x_train, y_train)))  %>%
  dataset_shuffle(buffer_size = buffer_size) %>%
  dataset_batch(batch_size, drop_remainder = TRUE)

validation_dataset <-
  tensor_slices_dataset(keras_array(list(x_valid, y_valid))) %>%
  dataset_shuffle(buffer_size = buffer_size) %>%
  dataset_batch(batch_size, drop_remainder = TRUE)


# Attention encoder -------------------------------------------------------


attention_encoder <-
  function(gru_units,
           embedding_dim,
           src_vocab_size,
           name = NULL) {
    keras_model_custom(name = name, function(self) {
      self$embedding <-
        layer_embedding(input_dim = src_vocab_size,
                        output_dim = embedding_dim)
      self$gru <-
        layer_gru(
          units = gru_units,
          return_sequences = TRUE,
          return_state = TRUE
        )
      
      function(inputs, mask = NULL) {
        x <- inputs[[1]]
        hidden <- inputs[[2]]
        
        x <- self$embedding(x)
        c(output, state) %<-% self$gru(x, initial_state = hidden)
        
        list(output, state)
      }
    })
  }



# Attention decoder -------------------------------------------------------


attention_decoder <-
  function(object,
           gru_units,
           embedding_dim,
           target_vocab_size,
           name = NULL) {
    keras_model_custom(name = name, function(self) {
      self$gru <-
        layer_gru(
          units = gru_units,
          return_sequences = TRUE,
          return_state = TRUE
        )
      self$embedding <-
        layer_embedding(input_dim = target_vocab_size, output_dim = embedding_dim)
      gru_units <- gru_units
      self$fc <- layer_dense(units = target_vocab_size)
      self$W1 <- layer_dense(units = gru_units)
      self$W2 <- layer_dense(units = gru_units)
      self$V <- layer_dense(units = 1L)
      
      function(inputs, mask = NULL) {
        x <- inputs[[1]]
        hidden <- inputs[[2]]
        encoder_output <- inputs[[3]]
        
        hidden_with_time_axis <- k_expand_dims(hidden, 2)
        
        score <-
          self$V(k_tanh(
            self$W1(encoder_output) + self$W2(hidden_with_time_axis)
          ))
        
        attention_weights <- k_softmax(score, axis = 2)
        
        context_vector <- attention_weights * encoder_output
        context_vector <- k_sum(context_vector, axis = 2)
        
        x <- self$embedding(x)
        
        x <-
          k_concatenate(list(k_expand_dims(context_vector, 2), x), axis = 3)
        
        c(output, state) %<-% self$gru(x)
        
        output <- k_reshape(output, c(-1, gru_units))
        
        x <- self$fc(output)
        
        list(x, state, attention_weights)
        
      }
      
    })
  }


# The model ---------------------------------------------------------------

encoder <- attention_encoder(
  gru_units = gru_units,
  embedding_dim = embedding_dim,
  src_vocab_size = src_vocab_size
)

decoder <- attention_decoder(
  gru_units = gru_units,
  embedding_dim = embedding_dim,
  target_vocab_size = target_vocab_size
)

optimizer <- tf$optimizers$Adam()

cx_loss <- function(y_true, y_pred) {
  mask <- ifelse(y_true == 0L, 0, 1)
  loss <-
    tf$nn$sparse_softmax_cross_entropy_with_logits(labels = y_true,
                                                   logits = y_pred) * mask
  tf$reduce_mean(loss)
}



# Inference / translation functions ---------------------------------------
# they are appearing here already in the file because we want to watch how
# the network learns

evaluate <-
  function(sentence) {
    attention_matrix <-
      matrix(0, nrow = target_maxlen, ncol = src_maxlen)
    
    sentence <- preprocess_sentence(sentence)
    input <- sentence2digits(sentence, src_index)
    input <-
      pad_sequences(list(input), maxlen = src_maxlen,  padding = "post")
    input <- k_constant(input)
    
    result <- ""
    
    hidden <- k_zeros(c(1, gru_units))
    c(enc_output, enc_hidden) %<-% encoder(list(input, hidden))
    
    dec_hidden <- enc_hidden
    dec_input <-
      k_expand_dims(list(word2index("<start>", target_index)))
    
    for (t in seq_len(target_maxlen - 1)) {
      c(preds, dec_hidden, attention_weights) %<-%
        decoder(list(dec_input, dec_hidden, enc_output))
      attention_weights <- k_reshape(attention_weights, c(-1))
      attention_matrix[t,] <- attention_weights %>% as.double()
      
      pred_idx <-
        tf$compat$v1$multinomial(k_exp(preds), num_samples = 1L)[1, 1] %>% as.double()
      pred_word <- index2word(pred_idx, target_index)
      
      if (pred_word == '<stop>') {
        result <-
          paste0(result, pred_word)
        return (list(result, sentence, attention_matrix))
      } else {
        result <-
          paste0(result, pred_word, " ")
        dec_input <- k_expand_dims(list(pred_idx))
      }
    }
    list(str_trim(result), sentence, attention_matrix)
  }

plot_attention <-
  function(attention_matrix,
           words_sentence,
           words_result) {
    melted <- melt(attention_matrix)
    ggplot(data = melted, aes(
      x = factor(Var2),
      y = factor(Var1),
      fill = value
    )) +
      geom_tile() + scale_fill_viridis() + guides(fill = FALSE) +
      theme(axis.ticks = element_blank()) +
      xlab("") +
      ylab("") +
      scale_x_discrete(labels = words_sentence, position = "top") +
      scale_y_discrete(labels = words_result) +
      theme(aspect.ratio = 1)
  }


translate <- function(sentence) {
  c(result, sentence, attention_matrix) %<-% evaluate(sentence)
  print(paste0("Input: ",  sentence))
  print(paste0("Predicted translation: ", result))
  attention_matrix <-
    attention_matrix[1:length(str_split(result, " ")[[1]]),
                     1:length(str_split(sentence, " ")[[1]])]
  plot_attention(attention_matrix,
                 str_split(sentence, " ")[[1]],
                 str_split(result, " ")[[1]])
}

# Training loop -----------------------------------------------------------


n_epochs <- 50

encoder_init_hidden <- k_zeros(c(batch_size, gru_units))

for (epoch in seq_len(n_epochs)) {
  total_loss <- 0
  iteration <- 0
  
  iter <- make_iterator_one_shot(train_dataset)
  
  until_out_of_range({
    batch <- iterator_get_next(iter)
    loss <- 0
    x <- batch[[1]]
    y <- batch[[2]]
    iteration <- iteration + 1

    with(tf$GradientTape() %as% tape, {
      c(enc_output, enc_hidden) %<-% encoder(list(x, encoder_init_hidden))
      
      dec_hidden <- enc_hidden
      dec_input <-
        k_expand_dims(rep(list(
          word2index("<start>", target_index)
        ), batch_size))
      
      
      for (t in seq_len(target_maxlen - 1)) {
        c(preds, dec_hidden, weights) %<-%
          decoder(list(dec_input, dec_hidden, enc_output))
        loss <- loss + cx_loss(y[, t], preds)
        
        dec_input <- k_expand_dims(y[, t])
      }
    })
    total_loss <-
      total_loss + loss / k_cast_to_floatx(dim(y)[2])
    
    paste0(
      "Batch loss (epoch/batch): ",
      epoch,
      "/",
      iteration,
      ": ",
      (loss / k_cast_to_floatx(dim(y)[2])) %>% as.double() %>% round(4),
      "\n"
    ) %>% print()
    
    variables <- c(encoder$variables, decoder$variables)
    gradients <- tape$gradient(loss, variables)
    
    optimizer$apply_gradients(purrr::transpose(list(gradients, variables)))
    
  })
  
  paste0(
    "Total loss (epoch): ",
    epoch,
    ": ",
    (total_loss / k_cast_to_floatx(buffer_size)) %>% as.double() %>% round(4),
    "\n"
  ) %>% print()
  
  walk(train_sentences[1:5], function(pair)
    translate(pair[1]))
  walk(validation_sample, function(pair)
    translate(pair[1]))
}

# plot a mask
example_sentence <- train_sentences[[1]]
translate(example_sentence)
