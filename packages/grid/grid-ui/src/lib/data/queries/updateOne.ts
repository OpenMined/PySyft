import api from '@/utils/api-axios'

export type UpdateOneArgs<T> = {
  data: T
  id: string
  queryKeys: string[]
}

export async function updateOne<T>({data, id, queryKeys}: UpdateOneArgs<T>): Promise<T> {
  const [route] = queryKeys
  const res = await api.patch<T>(`${route}/${id}`, data)
  return res.data
}
