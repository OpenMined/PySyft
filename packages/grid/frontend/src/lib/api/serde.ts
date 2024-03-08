import { serde } from "$lib/client/jsserde"
import type { KyResponse } from "ky"

export async function deserialize(response: KyResponse) {
  const buffer = await response.arrayBuffer()
  return serde.deserialize(buffer)
}

export function serialize(data: any) {
  return serde.serialize(data)
}

export function deserializeSyftResponseObject(data: Uint8Array) {
  return serde.deserialize(data).data
}
