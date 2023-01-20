import api from '@/utils/api'

export type UpdateOneArgs<T> = {
  data: T
  id: string
  queryKeys: string[]
}

export async function updateOne<T>({
  data,
  id,
  queryKeys,
}: UpdateOneArgs<T>): Promise<T> {
  const [route] = queryKeys
  const res = (await api.patch(`${route}/${id}`, { json: data }).json()) as T
  return res
}
