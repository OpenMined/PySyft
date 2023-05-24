import ky from 'ky';
import { syftCall } from './syft-api-call';
import { API_BASE_URL } from '../constants';
import { deserialize } from './serde';
import { parse as uuidParse } from 'uuid';
import { UUID } from '../client/objects/uid';

export async function getMetadata() {
  try {
    const res = await ky.get(`${API_BASE_URL}/metadata_capnp`);
    const metadata = await deserialize(res);

    const nodeUIDString = metadata?.id?.value;
    const nodeIdHyphen = hyphenateUUIDv4(nodeUIDString);
    window.localStorage.setItem('metadata', JSON.stringify(metadata));
    window.localStorage.setItem('nodeId', nodeIdHyphen);

    return metadata;
  } catch (error) {
    // TODO: Log error in debug mode
    throw error;
  }
}

function hyphenateUUIDv4(uuid: string): string {
  return uuid.replace(
    /([a-z0-9]{8})([a-z0-9]{4})([a-z0-9]{4})([a-z0-9]{4})([a-z0-9]{12})/,
    '$1-$2-$3-$4-$5'
  );
}

export async function updateMetadata(newMetadata) {
  const payload = {
    settings: { ...newMetadata, fqn: 'syft.service.settings.settings.NodeSettingsUpdate' }
  };
  return await syftCall({ path: 'settings.update', payload });
}
