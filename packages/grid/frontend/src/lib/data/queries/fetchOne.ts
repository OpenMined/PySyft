import api from '@/utils/api-axios'
import type {QueryFunctionContext} from 'react-query'

export type FetchOneContext = [string, string, string | number]

export async function fetchOne<T>({queryKey}: QueryFunctionContext<FetchOneContext>): Promise<T> {
  const [id, route] = queryKey
  const res = await api.get<T>(`${route}/${id}`)
  return res.data
}
