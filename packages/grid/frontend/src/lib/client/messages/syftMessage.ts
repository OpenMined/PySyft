import { UUID } from "../objects/uid.js"
import { v4 as uuidv4 } from "uuid"
import { SyftVerifyKey } from "../objects/key.js"
import sodium from "libsodium-wrappers"

export class SignedAPICall {
  credentials: SyftVerifyKey
  signature: Uint8Array
  serialized_message: Uint8Array
  fqn: string

  constructor(serialized_msg, signature, credentials) {
    this.serialized_message = serialized_msg
    this.signature = signature
    this.credentials = new SyftVerifyKey(credentials)
    this.fqn = "syft.client.api.SignedSyftAPICall"
  }

  get valid() {
    return sodium.crypto_sign_verify_detached(
      this.signature,
      this.serialized_message,
      this.credentials
    )
  }

  message(serde) {
    return serde.deserialize(this.serialized_message)
  }
}

export class APICall {
  server_uid: UUID
  path: string
  args: object
  kwargs: object
  blocking: boolean

  constructor(id, path, args, kwargs, blocking = true) {
    this.server_uid = new UUID(id)
    this.path = path
    this.args = args
    if (kwargs) {
      this.kwargs = new Map(Object.entries(kwargs))
    } else {
      this.kwargs = new Map()
    }
    this.blocking = blocking
    this.fqn = "syft.client.api.SyftAPICall"
  }

  sign(key, serde) {
    const serialized_message = new Uint8Array(serde.serialize(this))
    const signature = sodium.crypto_sign_detached(
      serialized_message,
      key.privateKey
    )
    return new SignedAPICall(serialized_message, signature, key.publicKey)
  }
}

class SignedMessage {
  address: UUID
  id: UUID
  serialized_message: Uint8Array
  signature: Uint8Array
  verify_key: Uint8Array

  constructor(
    address: UUID,
    msg: Uint8Array,
    private_key: Uint8Array,
    verify_key: Uint8Array
  ) {
    this.serialized_message = msg
    this.id = new UUID(uuidv4())
    this.address = address
    this.verify_key = verify_key
    this.signature = sodium.crypto_sign_detached(
      this.serialized_message,
      private_key
    )
  }

  message(serde) {
    return serde.deserialize(this.message)
  }
}

export class SyftMessage {
  address: UUID
  id: UUID
  reply_to: UUID
  reply: boolean
  kwargs: object
  fqn: string

  constructor(
    address: string,
    reply_to: string,
    reply: boolean,
    kwargs: object,
    fqn: string
  ) {
    this.address = new UUID(address)
    this.id = new UUID(uuidv4())
    this.reply_to = new UUID(reply_to)
    this.reply = reply
    this.kwargs = new Map(Object.entries(kwargs))
    this.fqn = fqn
  }

  sign(serde, private_key, verify_key) {
    const serialized_message = new Uint8Array(serde.serialize(this))
    return new SignedMessage(
      this.address,
      serialized_message,
      private_key,
      verify_key
    )
  }
}

export class SyftMessageWithReply extends SyftMessage {
  constructor(address: string, reply_to: string, kwargs: object, fqn: string) {
    super(address, reply_to, true, kwargs, fqn)
  }
}

export class SyftMessageWithoutReply extends SyftMessage {
  constructor(address: string, kwargs: object, fqn: string) {
    super(address, address, false, kwargs, fqn)
  }
}
