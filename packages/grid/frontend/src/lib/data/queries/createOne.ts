import api from '@/utils/api'

export type CreateArgs<T> = {
  data: T
  options: object & { multipart?: boolean }
  queryKeys: [string, string]
}

export async function createOne<T>({ queryKeys, data, options = {} }: CreateArgs<T>): Promise<T> {
  const [route] = queryKeys
  const { multipart, ...dataOptions } = options
  const res = multipart
    ? // @ts-ignore
      ((await api.post(route, { body: data, ...dataOptions }).json()) as T)
    : ((await api.post(route, { json: data, ...dataOptions }).json()) as T)
  return res
}
