import { makeSyftUID, syftCall } from './syft-api-call';

export async function getAllDatasets(page_size: number = 0, page_index: number = 0) {
  return await syftCall({
    path: 'dataset.get_all',
    payload: { page_size: page_size, page_index: page_index }
  });
}

export async function getDataset(uid: string) {
  return await syftCall({ path: 'dataset.get_by_id', payload: { uid: makeSyftUID(uid) } });
}

export async function deleteDataset(uid: string) {
  return await syftCall({ path: 'dataset.delete_by_id', payload: { uid: makeSyftUID(uid) } });
}

export async function searchDataset(name: string, chunk_size: number = 0, chunk_index: number = 0) {
  return await syftCall({
    path: 'dataset.search',
    payload: { name: name, chunk_size: chunk_size, chunk_index: chunk_index }
  });
}
