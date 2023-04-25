import ky from 'ky';
import { deserialize } from './serde';
// import { API_BASE_URL } from '../constants';
const API_BASE_URL = "api/v1";

export async function getMetadata() {
  try {
    const res = await ky.get(`${API_BASE_URL}/new/metadata_capnp`);

    const metadata = await deserialize(res);

    window.localStorage.setItem('metadata', JSON.stringify(metadata));
    window.localStorage.setItem('nodeId', metadata?.id?.value);

    return metadata;
  } catch (error) {
    // TODO: Log error in debug mode
    throw error;
  }
}
