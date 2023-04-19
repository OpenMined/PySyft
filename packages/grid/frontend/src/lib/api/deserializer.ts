import { serde } from '$lib/client/jsserde';
import type { KyResponse } from 'ky';

export async function deserialize(response: KyResponse) {
  const buffer = await response.arrayBuffer();
  return serde.deserialize(buffer);
}
