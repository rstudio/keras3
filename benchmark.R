
library(rlang)

.list =    \(...) list(...)
.list2 =   \(...) list2(...)
.splice =  \(x) list2(!!!x)
.do.call = \(x) do.call(list2, x)

x <- list("a", "b", "c")
print(r <- bench::mark(
  # list =    .list("a", "b", "c"),
  # list2 =   .list2("a", "b", "c"),
  splice =  .list2(!!!x),
  do.call = .do.call(x)
)); print(plot(r))
plot(r)
