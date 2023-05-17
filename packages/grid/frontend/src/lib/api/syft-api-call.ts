import ky from 'ky';
import sodium from 'libsodium-wrappers';
import { getKeyFromStorage, getNodeIdFromStorage } from './keys';
import { deserialize, deserializeSyftResponseObject, serialize } from './serde';
import { API_BASE_URL } from '../constants';

const SYFT_MSG_URL = `${API_BASE_URL}/new/api_call`;

export interface VerifyKey {
  key: Uint8Array;
  fqn: 'nacl.signing.VerifyKey';
}

export interface SyftCredentials {
  verify_key: VerifyKey;
  fqn: 'syft.node.credentials.SyftVerifyKey';
}

interface SyftCall {
  serializedMessage: Uint8Array;
  signature: Uint8Array;
  publicKey: Uint8Array;
}

interface SignedSyftCall {
  serialized_message: Uint8Array;
  signature: Uint8Array;
  credentials: SyftCredentials;
}

interface SyftUUID {
  value: string;
  fqn: 'syft.types.uid.UID';
}

interface SyftAPICall {
  path: string;
  payload: any;
}

export function getSyftNodeUID(): SyftUUID {
  return {
    value: getNodeIdFromStorage(),
    fqn: 'syft.types.uid.UID'
  };
}

export function makeSyftUID(uid: string): SyftUUID {
  return {
    value: uid,
    fqn: 'syft.types.uid.UID'
  };
}

export function isValid(syftCall: SignedSyftCall) {
  return sodium.crypto_sign_verify_detached(
    syftCall.signature,
    syftCall.serialized_message,
    syftCall.credentials.verify_key.key
  );
}

function prepareSyftCall(syftCall: SyftCall): SignedSyftCall {
  return {
    serialized_message: syftCall.serializedMessage,
    signature: syftCall.signature,
    credentials: {
      verify_key: {
        key: syftCall.publicKey,
        fqn: 'nacl.signing.VerifyKey'
      },
      fqn: 'syft.node.credentials.SyftVerifyKey'
    }
  };
}

export function signSyftMessage(apiCall: SyftAPICall) {
  try {
    const syftAPIPayload = {
      node_uid: getSyftNodeUID(),
      path: apiCall.path,
      args: [], // TODO: Check if this needs to be sent
      kwargs: new Map(Object.entries(apiCall.payload)),
      blocking: true,
      fqn: 'syft.client.api.SyftAPICall'
    };

    const serializedMessage = new Uint8Array(serialize(syftAPIPayload));

    const key = getKeyFromStorage();
    const signature = sodium.crypto_sign_detached(serializedMessage, key.privateKey);
    const signedMessage = prepareSyftCall({
      serializedMessage,
      signature,
      publicKey: key.publicKey
    });

    return {
      ...signedMessage,
      fqn: 'syft.client.api.SignedSyftAPICall'
    };
  } catch (error) {
    console.log(error);
    throw error;
  }
}

export async function syftCall({ path, payload }: SyftAPICall) {
  const signedMessage = signSyftMessage({ path, payload });
  const serializedMessage = serialize(signedMessage);

  const res = await ky.post(SYFT_MSG_URL, {
    headers: { 'content-type': 'application/octet-stream' },
    body: serializedMessage
  });

  const data = await deserialize(res);

  if (!isValid(data)) throw new Error('Invalid signature');

  return deserializeSyftResponseObject(data.serialized_message);
}
