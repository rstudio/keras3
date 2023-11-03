Callback used to stream events to a server.

@description
Requires the `requests` library.
Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
HTTP POST, with a `data` argument which is a
JSON-encoded dictionary of event data.
If `send_as_json=True`, the content type of the request will be
`"application/json"`.
Otherwise the serialized JSON will be sent within a form.

@param root String; root url of the target server.
@param path String; path relative to `root` to which the events will be sent.
@param field String; JSON field under which the data will be stored.
    The field is used only if the payload is sent within a form
    (i.e. send_as_json is set to False).
@param headers Dictionary; optional custom HTTP headers.
@param send_as_json Boolean; whether the request should be
    sent as `"application/json"`.

@export
@family callback
@seealso
+ <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/RemoteMonitor>
