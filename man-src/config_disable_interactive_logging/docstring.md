Turn off interactive logging.

When interactive logging is disabled, Keras sends logs to `absl.logging`.
This is the best option when using Keras in a non-interactive
way, such as running a training or inference job on a server.
