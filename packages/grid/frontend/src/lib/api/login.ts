import ky from 'ky';
import { deserialize } from './deserializer';
import { API_BASE_URL } from '../constants';

interface Credentials {
  email: string;
  password: string;
}

export async function login({ email, password }: Credentials) {
  try {
    const res = await ky.post(`${API_BASE_URL}/new/login`, {
      json: { email, password }
    });

    const data = await deserialize(res);
    const signing_key = data.signing_key.signing_key;

    window.localStorage.setItem('key', signing_key);
    window.localStorage.setItem('id', data.id.value);
  } catch (error) {
    console.log(error);
  }
}
