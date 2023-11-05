callback_remote_monitor <-
function (root = "http://localhost:9000", path = "/publish/epoch/end/", 
    field = "data", headers = NULL, send_as_json = FALSE) 
{
    args <- capture_args2(NULL)
    do.call(keras$callbacks$RemoteMonitor, args)
}
