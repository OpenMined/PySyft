import { getUserIdFromStorage } from './keys';
import { makeSyftUID, syftCall } from './syft-api-call';

export async function getAllUsers(page_size: number = 0, page_index: number = 0) {
  return await syftCall({
    path: 'user.get_all',
    payload: { page_size: page_size, page_index: page_index }
  });
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

export async function searchUsersByName(
  name: string,
  page_size: number = 0,
  page_index: number = 0
) {
  return await syftCall({
    path: 'user.search',
    payload: {
      user_search: { name: name, fqn: 'syft.service.user.user.UserSearch' },
      page_size: page_size,
      page_index: page_index
    }
  });
}

export async function updateCurrentUser(
  name: string,
  email: string,
  password: string,
  institution: string,
  website: string
) {
  const userUpdate = {
    name: name,
    email: email,
    password: password,
    institution: institution,
    website: website,
    fqn: 'syft.service.user.user.UserUpdate'
  };

  return await syftCall({
    path: 'user.update',
    payload: {
      uid: makeSyftUID(getUserIdFromStorage()),
      user_update: userUpdate
    }
  });
}
