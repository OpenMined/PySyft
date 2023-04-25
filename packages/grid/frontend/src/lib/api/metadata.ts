import ky from 'ky';
import { deserialize } from './serde';
import { syftCall } from './syft-api-call';
import { API_BASE_URL } from '../constants';

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

export async function updateMetadata(newMetadata) {
  const payload = {
    metadata: { ...newMetadata, fqn: 'syft.service.metadata.node_metadata.NodeMetadataUpdate' }
  };
  return await syftCall({ path: 'metadata.update', payload });
}
