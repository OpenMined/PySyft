import { makeSyftUID, syftCall } from './syft-api-call';

export async function getAllCodeRequests() {
  return await syftCall({ path: 'code.get_all', payload: {} });
}

export async function getAllRequests() {
  return await syftCall({ path: 'request.get_all_info', payload: {} });
}

export async function filterRequests(name: string, page_index: number = 0, page_size: number = 0) {
  return await syftCall({
    path: 'request.filter_all_info',
    payload: {
      request_filter: { name: name, fqn: 'syft.service.request.request.RequestInfoFilter' },
      page_size: page_size,
      page_index: page_index
    }
  });
}

export async function getCodeRequest(uid: string) {
  return await syftCall({ path: 'code.get_by_id', payload: { uid: makeSyftUID(uid) } });
}
