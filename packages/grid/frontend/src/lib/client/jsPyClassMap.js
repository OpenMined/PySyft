import { SyftVerifyKey, VerifyKey } from '$lib/client/objects/key.ts';
import { UUID } from '$lib/client/objects/uid.ts';

export const classMapping = {
  'syft.node.credentials.SyftVerifyKey': SyftVerifyKey,
  'nacl.signing.VerifyKey': VerifyKey,
  'syft.types.uid.UID': UUID
};
