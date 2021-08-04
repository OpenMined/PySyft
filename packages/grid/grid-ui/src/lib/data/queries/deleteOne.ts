import api from '@/utils/api-axios'

export type DeleteOneArgs<T> = {
  data: T
  id: string | number
  queryKeys: string[]
}

export async function deleteOne<T>({id, queryKeys}: DeleteOneArgs<T>) {
  const [route] = queryKeys
  const res = await api.delete(`${route}/${id}`)
  return res.data
}
