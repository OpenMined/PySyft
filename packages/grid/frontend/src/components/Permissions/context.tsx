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
}

const PermissionContext = createContext<PermissionContextProps>({
  role: null,
  permissions: null,
  save: null,
  toggle: null,
})

function PermissionsAccordionProvider({ role, children }) {
  const [permissions, setPermissions] = useState<AllSyftPermissions>(() => {
    const { id, name, ...allPermissions } = role
    return allPermissions
  })

  const update = useRoles().update(role.id).mutate

  const toggle = (permission: SyftPermissions) =>
    setPermissions((prevPermissions) => ({
      ...prevPermissions,
      [permission]: !prevPermissions[permission],
    }))

  const save = () => update({ ...role, ...permissions })

  return (
    <PermissionContext.Provider value={{ permissions, role, save, toggle }}>
      {children}
    </PermissionContext.Provider>
  )
}

const usePermission = () => useContext(PermissionContext)

export { PermissionsAccordionProvider, usePermission }
