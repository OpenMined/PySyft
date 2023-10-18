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

type SyftCredentials = {
  verify_key: {
    key: Uint8Array
    fqn: string
  }
  fqn: string
}

interface SignedSyftCall {
  serialized_message: Uint8Array
  signature: Uint8Array
  credentials: SyftCredentials
  fqn: string
}

interface SyftAPICall {
  path: string
  payload: Record<string, any>
  signing_key: string | Uint8Array
  node_id: string
}

export const isValid = (syftCall: SignedSyftCall): boolean => {
  const {
    signature,
    serialized_message,
    credentials: {
      verify_key: { key },
    },
  } = syftCall
  return sodium.crypto_sign_verify_detached(signature, serialized_message, key)
}

export const sign_message = ({
  path,
  payload,
  signing_key,
  node_id,
}: SyftAPICall): SignedSyftCall => {
  if (typeof signing_key === "string") throw new Error("Invalid key")

  try {
    const syftAPIPayload = {
      path,
      node_uid: {
        value: node_id,
        fqn: FQN.UID,
      },
      blocking: true,
      args: [], // TODO: Check if this needs to be sent
      kwargs: new Map(Object.entries(payload)),
      fqn: FQN.SYFT_API_CALL,
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
        verify_key: {
          key: signing_key.publicKey,
          fqn: FQN.VERIFY_KEY,
        },
        fqn: FQN.SYFT_VERIFY_KEY,
      },
      fqn: FQN.SIGNED_SYFT_API_CALL,
    }
  } catch (error) {
    console.error(error)
    throw error
  }
}

export const js_syft_call = async ({
  path,
  payload,
  signing_key,
  node_id,
}: SyftAPICall): Promise<any> => {
  let key = signing_key

  if (typeof key === "string") {
    key = Uint8Array.from(key.split(","))
    key = sodium.crypto_sign_seed_keypair(key)
  }

  const signedMessage = sign_message({
    path,
    payload,
    signing_key: key,
    node_id,
  })

  try {
    const res = await ky.post(SYFT_MSG_URL, {
      headers: { "content-type": "application/octet-stream" },
      body: serialize(signedMessage),
    })

    const data = await deserialize(res)

    if (!isValid(data)) throw new Error("Invalid signature")

    return deserializeSyftResponseObject(data.serialized_message)
  } catch (error) {
    console.error(error?.message || error)
    throw error
  }
}
