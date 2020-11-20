
PROTOCOL_URI = "did:sov:BzCbsNYhMrjHiqZDTUASHg;spec/protocolexample/1.0"

PROTOCOL_EXAMPLE = f"{PROTOCOL_URI}/protocolexample"
PROTOCOL_EXAMPLE_RESPONSE = f"{PROTOCOL_URI}/protocolexample_response"

NEW_PROTOCOL_URI = "https://didcomm.org/protocolexample/1.0"

NEW_PPML = f"{NEW_PROTOCOL_URI}/protocolexample"
NEW_PPML_RESPONSE = f"{NEW_PROTOCOL_URI}/protocolexample_response"

PROTOCOL_PACKAGE = "acapy_protocol_example.protocolexample.v1_0"

MESSAGE_TYPES = {
    PROTOCOL_EXAMPLE: f"{PROTOCOL_PACKAGE}.messages.protocolexample.ProtocolExample",
    PROTOCOL_EXAMPLE_RESPONSE: f"{PROTOCOL_PACKAGE}.messages.protocolexample_response.ProtocolExampleResponse",
    NEW_PPML: f"{PROTOCOL_PACKAGE}.messages.protocolexample.ProtocolExample",
    NEW_PPML_RESPONSE: f"{PROTOCOL_PACKAGE}.messages.protocolexample_response.ProtocolExampleResponse",
}
