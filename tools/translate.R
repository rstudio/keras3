
envir::attach_source("tools/setup.R")
envir::attach_source("tools/common.R")

# py_install("openai")
use_codellama <- FALSE
if (use_codellama) {
  Sys.unsetenv("OPENAI_API_KEY")
  openai <- import("openai")
  openai$api_base <- glue("http://192.168.1.15:9092")

} else {
  readRenviron("~/.Renviron")
  import("os")$environ$update(list(OPENAI_API_KEY = Sys.getenv("OPENAI_API_KEY")))
  openai <- import("openai")
}

# use like msg(user = ...), assistant =
chat_messages <- function(...) {
  x <- rlang::dots_list(..., .named = TRUE, .ignore_empty = "all")
  stopifnot(all(names(x) %in% c("system", "user", "assistant")))
  unname(imap(x, \(content, role) lst(role, content = str_flatten_lines(content))))
}

# if(FALSE) {
#
# models <- openai$Model$list()
#
# models[1:2] %>% str()
#
# map_chr(models$data, "id") %>%
#   grep("4", ., value = T)
# }

system.time(chat_completion <- openai$ChatCompletion$create(
  model="gpt-4",
  # model="gpt-4-32k"
  # model="gpt-3.5-turbo",
  # model="gpt-3.5-turbo-16k",
  temperature = 0,
  messages = chat_messages %(% {

    system = "Translate Python to R, leave everthing else the same."

    ###
    user = "x = tf.random.normal(input_shape)"
    assistant = "x <- tf$random$normal(input_shape)"

    ###
    user = r"--(
    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.
    )--"

    assistant = r"--(
    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is `TRUE`,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `NULL`, it is applied to the outputs as well.
    )--"

    ###
    user = r"--(
    # The inputs are 28x28 RGB images with `channels_last` and the batch
    # size is 4.
    input_shape = (4, 28, 28, 3)
    x = tf.random.normal(input_shape)
    y = tf.keras.layers.Conv2D(
    2, 3, activation='relu', input_shape=input_shape[1:])(x)
    print(y.shape)
    # (4, 26, 26, 2)
    )--"

    assistant = r"--(
    # The inputs are 28x28 RGB images with `channels_last` and the batch
    # size is 4.
    input_shape = shape(4, 28, 28, 3)
    x <- tf$random$normal(input_shape)
    y <- x |>
      layer_conv_2d(2, 3, activation='relu')
    y$shape
    # (4, 26, 26, 2)
    )--"

    ###
    # user = r"--(
    # # With `dilation_rate` as 2.
    # input_shape = (4, 28, 28, 3)
    # x = tf.random.normal(input_shape)
    # y = tf.keras.layers.Conv2D(
    #     2, 3,
    #     activation='relu',
    #     dilation_rate=2,
    #     input_shape=input_shape[1:])(x)
    # print(y.shape)
    # # (4, 24, 24, 2)
    # )--"
    # user = r"---(
    #
    # )---"
    user = readLines("tools/raw/keras-layers-Dense/r_wrapper.R")
  }
))
# 120 seconds w/ Conv2D for code llama
# 106 seconds w/ Conv2D for openai

chat_completion$choices[[1]]$message$content |> cat()


waldo::compare(
  str_flatten_lines(readLines("tools/raw/keras-layers-Dense/r_wrapper.R")),
  chat_completion$choices[[1]]$message$content)
