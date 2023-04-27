import ky from 'ky';
import { deserialize, serialize } from './serde';
import { API_BASE_URL } from '../constants';

interface LoginCredentials {
  email: string;
  password: string;
}

interface SignUpDetails {
  email: string;
  password: string;
  password_verify: string;
  name: string;
  institution?: string;
  website?: string;
  role: number;
}

export async function login({ email, password }: LoginCredentials) {
  const res = await ky.post(`${API_BASE_URL}/new/login`, {
    json: { email, password }
  });

  const data = await deserialize(res);

  const signing_key = data.signing_key.signing_key;

  window.localStorage.setItem('key', signing_key);
  window.localStorage.setItem('id', data.id.value);

  return data;
}

export async function register(newUser: SignUpDetails) {
  const { email, password, password_verify, name } = newUser;

  if (!email || !password || !password_verify || !name) {
    throw new Error('Missing required fields');
  }

  const payload = serialize({
    ...newUser,
    fqn: 'syft.service.user.user.UserCreate'
  });

  const res = await ky.post(`${API_BASE_URL}/new/register`, {
    headers: { 'content-type': 'application/octet-stream' },
    body: payload
  });

  const data = await deserialize(res);

  if (Array.isArray(data)) {
    return data;
  }

  throw new Error('Unexpected response');
}
