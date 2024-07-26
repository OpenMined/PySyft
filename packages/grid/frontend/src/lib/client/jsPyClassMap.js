import { SyftVerifyKey, VerifyKey } from "$lib/client/objects/key.ts"
import { UUID } from "$lib/client/objects/uid.ts"
import { APICall } from "$lib/client/messages/syftMessage.ts"

export const classMapping = {
  "syft.server.credentials.SyftVerifyKey": SyftVerifyKey,
  "nacl.signing.VerifyKey": VerifyKey,
  "syft.types.uid.UID": UUID,
  "syft.client.api.SyftAPICall": APICall,
}
