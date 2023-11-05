callback_csv_logger <-
function (filename, separator = ",", append = FALSE) 
{
    args <- capture_args2(NULL)
    do.call(keras$callbacks$CSVLogger, args)
}
