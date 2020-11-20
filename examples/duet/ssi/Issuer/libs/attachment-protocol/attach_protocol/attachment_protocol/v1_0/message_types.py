
PROTOCOL_URI ="did:sov:BzCbsNYhMrjHiqZDTUASHg;spec/attachmentprotocol/1.0"

ATTACHMENT = f"{PROTOCOL_URI}/attachmentprotocol"

NEW_PROTOCOL_URI ="https://didcomm.org/attachmentprotocol/1.0"

NEW_ATTACHMENT_PROTOCOL = f"{NEW_PROTOCOL_URI}/attachmentprotocol"

PROTOCOL_PACKAGE ="attach_protocol.attachment_protocol.v1_0"

MESSAGE_TYPES = {
    ATTACHMENT: f"{PROTOCOL_PACKAGE}.messages.attachment.Attachment",
    NEW_ATTACHMENT_PROTOCOL: f"{PROTOCOL_PACKAGE}.messages.attachment.Attachment",
}