import ky from "ky"
import sodium from "libsodium-wrappers"
import { deserialize, deserializeSyftResponseObject, serialize } from "./serde"
import { API_BASE_URL } from "../constants"

const SYFT_MSG_URL = `${API_BASE_URL}/api_call`

const FQN = {
  VERIFY_KEY: "nacl.signing.VerifyKey",
  SYFT_VERIFY_KEY: "syft.node.credentials.SyftVerifyKey",
  UID: "syft.types.uid.UID",
  SYFT_API_CALL: "syft.client.api.SyftAPICall",
  SIGNED_SYFT_API_CALL: "syft.client.api.SignedSyftAPICall",
}

interface SyftAPICall {
  path: string
  payload: Record<string, any>
  signing_key: string | Uint8Array
  node_id: string
}

const getKeyPair = (signing_key: string | Uint8Array) =>
  typeof signing_key === "string"
    ? sodium.crypto_sign_seed_keypair(Uint8Array.from(signing_key.split(",")))
    : signing_key

const getSignedMessage = ({
  path,
  payload,
  node_id,
  signing_key,
}: SyftAPICall) => {
  console.log(typeof signing_key)
  const syftAPIPayload = {
    path,
    node_uid: { value: node_id, fqn: FQN.UID },
    args: [],
    kwargs: new Map(Object.entries(payload)),
    fqn: FQN.SYFT_API_CALL,
    blocking: true,
  }

  const serializedMessage = new Uint8Array(serialize(syftAPIPayload))
  const signature = sodium.crypto_sign_detached(
    serializedMessage,
    signing_key.privateKey
  )

  return {
    serialized_message: serializedMessage,
    signature,
    credentials: {
      verify_key: { key: signing_key.publicKey, fqn: FQN.VERIFY_KEY },
      fqn: FQN.SYFT_VERIFY_KEY,
    },
    fqn: FQN.SIGNED_SYFT_API_CALL,
  }
}

const send = async (signedMessage: any): Promise<any> => {
  try {
    const res = await ky.post(SYFT_MSG_URL, {
      headers: { "content-type": "application/octet-stream" },
      body: serialize(signedMessage),
    })

    const data = await deserialize(res)

    if (
      !sodium.crypto_sign_verify_detached(
        signedMessage.signature,
        signedMessage.serialized_message,
        signedMessage.credentials.verify_key.key
      )
    ) {
      throw new Error("Invalid signature")
    }

    return deserializeSyftResponseObject(data.serialized_message)
  } catch (error) {
    console.error(error?.message || error)
    throw error
  }
}

export const jsSyftCall = async ({
  path,
  payload,
  signing_key,
  node_id,
}: SyftAPICall): Promise<any> => {
  const key = getKeyPair(signing_key)
  const signedMessage = getSignedMessage({
    path,
    payload,
    node_id,
    signing_key: key,
  })
  return await send(signedMessage)
}
