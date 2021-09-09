import api from '@/utils/api-axios'
import {AxiosRequestConfig} from 'axios'

export type CreateArgs<T> = {
  data: T
  options: AxiosRequestConfig
  queryKeys: [string, string]
}

export async function createOne<T>({queryKeys, data, options = {}}: CreateArgs<T>): Promise<T> {
  const [route] = queryKeys
  const res = await api.post<T>(route, data, options)
  return res.data
}
