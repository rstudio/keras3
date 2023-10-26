


# for(dir in list.dirs("tools/raw"))
list.dirs("tools/raw") %>%
  lapply(function(d) {
    withr::local_dir(d)
    if(any(startsWith(dir(), "gpt")))
      system(paste("code -d r_wrapper.R", dir(pattern = "^gpt-4")[1]))
  }) %>%
  invisible()
