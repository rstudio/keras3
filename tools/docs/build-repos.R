# build repository
pkg <- devtools::build()
oldwd <- setwd("docs/repos")
drat.builder::build()
drat::insertPackage(pkg, repodir = ".")
setwd(oldwd)


