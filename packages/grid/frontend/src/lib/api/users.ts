import { getUserIdFromStorage } from './keys';
import { makeSyftUID, syftCall } from './syft-api-call';
import { SyftSigningKey } from '$lib/client/objects/key';

interface SignUpDetails {
  email: string;
  password: string;
  password_verify: string;
  name: string;
  institution?: string;
  website?: string;
  role: number;
}

export async function getAllUsers(page_size: number = 0, page_index: number = 0) {
  return await syftCall({
    path: 'user.get_all',
    payload: { page_size: page_size, page_index: page_index }
  });
}

export async function getUser(uid: string) {
  return await syftCall({ path: 'user.view', payload: { uid: makeSyftUID(uid) } });
}

export async function createUser(newUser: SignUpDetails) {
  const { email, password, password_verify, name } = newUser;

  if (!email || !password || !password_verify || !name) {
    throw new Error('Missing required fields');
  }

  const key: string = window.localStorage.getItem('key') || '';
  let signingKey: SyftSigningKey | undefined = undefined;
  if (key) {
    let signingKey = new SyftSigningKey(Uint8Array.from(key.split(',')));
  }
  const createUser = {
    ...newUser,
    created_by: signingKey,
    fqn: 'syft.service.user.user.UserCreate'
  };
  return await syftCall({ path: 'user.create', payload: { user_create: createUser } });
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
