import sodium from 'libsodium-wrappers';

export function getKeyFromStorage() {
  let key = window.localStorage.getItem('key');

  if (!key) throw new Error('Key not found');

  try {
    key = Uint8Array.from(key.split(','));
    return sodium.crypto_sign_seed_keypair(key);
  } catch (error) {
    throw new Error('Invalid key');
  }
}

export function getNodeIdFromStorage() {
  const key = window.localStorage.getItem('nodeId');

  if (!key) throw new Error('Node ID not found');

  return key;
}

export function getUserIdFromStorage() {
  const key = window.localStorage.getItem('id');

  if (!key) throw new Error('User ID not found');

  return key;
}
