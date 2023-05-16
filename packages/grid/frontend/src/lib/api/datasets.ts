import { makeSyftUID, syftCall } from './syft-api-call';

export async function getAllDatasets() {
  return await syftCall({ path: 'dataset.get_all', payload: {} });
}

export async function getDataset(uid: string) {
  return await syftCall({ path: 'dataset.get_by_id', payload: { uid: makeSyftUID(uid) } });
}

export async function deleteDataset(uid: string) {
  return await syftCall({ path: 'dataset.delete_by_id', payload: { uid: makeSyftUID(uid) } });
}
