callback_backup_and_restore <-
function (backup_dir, save_freq = "epoch", delete_checkpoint = TRUE) 
{
    args <- capture_args2(list(save_freq = as_integer))
    do.call(keras$callbacks$BackupAndRestore, args)
}
