import { getUserIdFromStorage } from './keys';
import { makeSyftUID, syftCall } from './syft-api-call';

export async function getAllUsers() {
  return await syftCall({ path: 'user.get_all', payload: {} });
}

export async function getUser(uid: string) {
  return await syftCall({ path: 'user.view', payload: { uid: makeSyftUID(uid) } });
}

export async function getSelf() {
  return await syftCall({
    path: 'user.view',
    payload: { uid: makeSyftUID(getUserIdFromStorage()) }
  });
}

export async function searchUsersByName(name: string) {
  return await syftCall({ path: 'user.search', payload: { name } });
}
