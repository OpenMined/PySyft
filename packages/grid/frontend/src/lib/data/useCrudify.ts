import { useQuery, useMutation, useQueryClient } from 'react-query'
import { fetchAll, fetchOne, createOne, updateOne, deleteOne } from './queries'
import type {
  UseQueryOptions,
  QueryFunction,
  MutationFunction,
  UseMutationOptions,
} from 'react-query'
import type { AxiosRequestConfig } from 'axios'
import type { ErrorMessage } from '@/utils/api-axios'

interface Crudify<T> {
  queryKeys: string[]
  invalidateKeys?: string[]
  fetchAllFn: QueryFunction<T[]>
  fetchOneFn: QueryFunction<T>
  createFn: MutationFunction<T>
  updateFn: MutationFunction<T>
  deleteFn: MutationFunction<T>
}

const invalidateQueriesConfig = {
  refetchInactive: true,
}

function useCrudifyRQ<T>({
  queryKeys,
  invalidateKeys,
  fetchAllFn,
  fetchOneFn,
  createFn,
  updateFn,
  deleteFn,
}: Crudify<T>) {
  const queryClient = useQueryClient()

  const useAll = (config: UseQueryOptions<T[]> = {}) =>
    useQuery<T[], ErrorMessage>(queryKeys, fetchAllFn, config)

  const useOne = (id: string | number, config?: UseQueryOptions<T>) =>
    useQuery<T, ErrorMessage>([id, ...queryKeys], fetchOneFn, {
      ...config,
      enabled: !!id,
    })

  const useCreate = <U>(
    config: UseMutationOptions<T, ErrorMessage, U> = {},
    options?: AxiosRequestConfig
  ) =>
    useMutation<T, ErrorMessage, U>(
      (data) => createFn({ data, queryKeys, options }),
      {
        ...config,
        onSuccess: (...args) => {
          queryClient.invalidateQueries(queryKeys, invalidateQueriesConfig)
          queryClient.invalidateQueries(invalidateKeys, invalidateQueriesConfig)
          if (config.onSuccess) {
            config.onSuccess(...args)
          }
        },
      }
    )

  const useUpdate = <U>(
    id: string | number,
    config: UseMutationOptions<T, ErrorMessage, U> = {}
  ) =>
    useMutation<T, ErrorMessage, U>(
      (data) => updateFn({ data, id, queryKeys }),
      {
        ...config,
        onSuccess: (...args) => {
          queryClient.invalidateQueries(
            [id, ...queryKeys],
            invalidateQueriesConfig
          )
          queryClient.invalidateQueries(queryKeys, invalidateQueriesConfig)
          queryClient.invalidateQueries(invalidateKeys, invalidateQueriesConfig)
          if (config.onSuccess) {
            config.onSuccess(...args)
          }
        },
      }
    )

  const useDelete = <U>(
    id: string | number,
    config: UseMutationOptions<T, ErrorMessage, void, U> = {}
  ) =>
    useMutation<T, ErrorMessage, void, U>(() => deleteFn({ id, queryKeys }), {
      ...config,
      onSuccess: (...args) => {
        queryClient.invalidateQueries(queryKeys, invalidateQueriesConfig)
        queryClient.invalidateQueries(invalidateKeys, invalidateQueriesConfig)
        if (config.onSuccess) {
          config.onSuccess(...args)
        }
      },
    })

  return {
    all: useAll,
    one: useOne,
    create: useCreate,
    update: useUpdate,
    remove: useDelete,
  }
}

export function useCrudify<T>(queryKeys: string[], invalidateKeys?: string[]) {
  return useCrudifyRQ<T>({
    queryKeys,
    invalidateKeys,
    fetchAllFn: fetchAll,
    fetchOneFn: fetchOne,
    createFn: createOne,
    updateFn: updateOne,
    deleteFn: deleteOne,
  })
}
