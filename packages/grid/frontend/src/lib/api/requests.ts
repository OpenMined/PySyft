import { makeSyftUID, syftCall } from './syft-api-call';

export async function getAllCodeRequests() {
  return await syftCall({ path: 'code.get_all', payload: {} });
}

export async function getCodeRequest(uid: string) {
  return await syftCall({ path: 'code.get_by_id', payload: { uid: makeSyftUID(uid) } });
}
