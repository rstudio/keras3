

if(!"source:tools/utils.R" %in% search()) envir::attach_source("tools/utils.R")

missing_returns_section <- function(block) {
  # tags <- lapply(block$tags, \(t) t$tag)
  # !any(c("returns", "return") %in% tags)
  !roxygen2::block_has_tags(block, c("return", "returns"))
}

# Missing @returns
walk_roxy_blocks(function(block) {
  if(roxygen2::block_has_tags(block, "noRd"))
    return()
  if(missing_returns_section(block)) {
    if("return" %in% tryCatch(
      roxygen2::block_get_tag_value(block, "inherit")$fields,
      error = \(e) ""))
      return() # taken care of already
    file <- block$file
    line <- block$line
    name <- doctether:::get_block_name(block)
    if (name |> startsWith("metric")) #browser()
    # if(identical(c("y_true", "y_pred") ,
    #              (block$object$value |> formals() |> names() |> _[1:2]))) {

      cli_alert_info("{name} {.file {file}:{line}}")
    # }
  }
})

