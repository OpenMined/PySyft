import api from '@/utils/api-axios'
import type {QueryFunctionContext} from 'react-query'

export type FetchAllContext = [string, string]

export async function fetchAll<T>({queryKey}: QueryFunctionContext<FetchAllContext>): Promise<T> {
  const [route] = queryKey
  const res = await api.get<T>(route)
  return res.data
}
