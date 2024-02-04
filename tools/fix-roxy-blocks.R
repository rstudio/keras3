

if(!"source:tools/utils.R" %in% search()) envir::attach_source("tools/utils.R")

missing_returns_section <- function(block) {
  tags <- lapply(block$tags, \(t) t$tag)
  !"returns" %in% tags
}

# Missing @returns
walk_roxy_blocks(function(block) {
  if(roxygen2::block_has_tags(block, "noRd"))
    return()
  if(missing_returns_section(block)) {
    file <- block$file
    line <- block$line
    name <- doctether:::get_block_name(block)
    cli_alert_info("{name} {.file {file}:{line}}")
  }
})
