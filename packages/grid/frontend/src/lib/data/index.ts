import {useQuery} from 'react-query'
import {cacheKeys} from '@/utils'
import {useCrudify} from '@/lib/data/useCrudify'
import type {User, UserMe, Dataset, Request, Tensor, Model, Role, Settings, DomainStatus} from '@/types/grid-types'

export function useMe() {
  return useQuery<UserMe>(cacheKeys.me)
}

export function useUsers() {
  return useCrudify<User>([cacheKeys.users])
}

export function useDatasets() {
  return useCrudify<Dataset>([cacheKeys.datasets])
}

export function useModels() {
  return useCrudify<Model>([cacheKeys.models])
}

export function useRequests() {
  return useCrudify<Request>([cacheKeys.requests])
}

export function useTensors() {
  return useCrudify<Tensor>([cacheKeys.tensors])
}

export function useRoles() {
  return useCrudify<Role>([cacheKeys.roles])
}

export function useSettings() {
  return useQuery<Settings>(cacheKeys.settings)
}

export function useDomainStatus() {
  return useQuery<DomainStatus>(cacheKeys.status)
}

export function useInitialSetup() {
  return useCrudify<Settings>([cacheKeys.settings]).create
}
