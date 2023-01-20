import { createContext, useContext, useState } from 'react'
import { useRoles } from '@/lib/data'

import type { UseMutateFunction } from 'react-query'
import type {
  Role,
  AllSyftPermissions,
  SyftPermissions,
} from '@/types/permissions'

interface PermissionContextProps {
  role: Role
  permissions: AllSyftPermissions
  toggle: (_: SyftPermissions) => void
  save: UseMutateFunction
  isSuccess: boolean
  isLoading: boolean
}

const PermissionContext = createContext<PermissionContextProps>({
  role: null,
  permissions: null,
  save: null,
  toggle: null,
  isSuccess: false,
  isLoading: false,
})

function PermissionsAccordionProvider({ role, children }) {
  const [permissions, setPermissions] = useState<AllSyftPermissions>(() => {
    const { id, name, ...allPermissions } = role
    return allPermissions
  })

  const permissionUpdate = useRoles().update(role.id)
  const { mutate, isLoading, isSuccess } = permissionUpdate

  const toggle = (permission: SyftPermissions) =>
    setPermissions((prevPermissions) => ({
      ...prevPermissions,
      [permission]: !prevPermissions[permission],
    }))

  const save = () => mutate({ ...role, ...permissions })

  return (
    <PermissionContext.Provider
      value={{ permissions, role, save, toggle, isLoading, isSuccess }}
    >
      {children}
    </PermissionContext.Provider>
  )
}

const usePermission = () => useContext(PermissionContext)

export { PermissionsAccordionProvider, usePermission }
