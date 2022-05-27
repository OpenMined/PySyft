import { useQuery } from 'react-query'
import { cacheKeys } from '@/utils'
import { useCrudify } from '@/lib/data/useCrudify'
import api from '@/utils/api'
import type {
  User,
  UserMe,
  Dataset,
  Request,
  Settings,
} from '@/types/grid-types'
import type { Role } from '@/types/permissions'

export function useMe() {
  return useQuery<UserMe>(
    cacheKeys.me,
    () => api.get(cacheKeys.me).json() as UserMe
  )
}

export function useAssociationRequest() {
  return useCrudify([cacheKeys.association_request])
}

export function useUsers() {
  return useCrudify<User>([cacheKeys.users], [cacheKeys.me])
}

export function useApplicantUsers() {
  return useCrudify<User>([cacheKeys.applicant_users], [cacheKeys.me])
}

export function useDatasets() {
  return useCrudify<Dataset>([cacheKeys.datasets])
}

export function useRequests() {
  return useCrudify<Request>([cacheKeys.requests])
}

export function useDataRequests() {
  return useCrudify<Request>([cacheKeys.data])
}

export function useBudgetRequests() {
  return useCrudify<Request>([cacheKeys.budget])
}

export function useRoles() {
  return useCrudify<Role>([cacheKeys.roles], [cacheKeys.users, cacheKeys.me])
}

export function useSettings() {
  return useCrudify<Settings>([cacheKeys.settings])
}
