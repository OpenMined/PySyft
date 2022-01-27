import { useState } from 'react'
import { NormalButton, DeleteButton, Accordion, PermissionsListToggle } from '@/components'
import { useRoles } from '@/lib/data'
import type { UserPermissions, Role } from '@/types/grid-types'

// TODO: Unfortunately Grid is not Restful (yet).
//
export function PermissionList({ roles }: { roles: Role[] }) {
  return (
    <Accordion>
      {roles.map(role => (
        <Accordion.Item key={role.id}>
          <PermissionInfoTitle name={role.name} />
          <PermissionInfoPanel {...role} />
        </Accordion.Item>
      ))}
    </Accordion>
  )
}

function PermissionInfoTitle({ name }: Partial<Role>) {
  return (
    <div className="flex space-x-2 truncate">
      <p className="font-medium truncate">{name}</p>
    </div>
  )
}

function DeleteRole({ id }: { id: string | number }) {
  const { remove } = useRoles()
  const mutation = remove(id)
  return (
    <DeleteButton
      isLoading={mutation.isLoading}
      onClick={() => mutation.mutate()}
      disabled={mutation.isLoading}
      className="w-full md:w-32"
    >
      Delete role
    </DeleteButton>
  )
}

function PermissionInfoPanel(role: Role) {
  const [editableRole, setRole] = useState<Role>(role)
  const { update } = useRoles()
  const mutation = update(role.id)

  const changePermission = (permission: UserPermissions, value: boolean) =>
    setRole(role => ({ ...role, [permission]: value }))

  return (
    <div className="px-16 py-6 space-y-6 text-sm border-t border-gray-200 bg-blueGray-100">
      <div className="space-y-6">
        <PermissionsListToggle
          id={role.id}
          defaultPermissions={editableRole}
          onChange={(permission: UserPermissions, enabled: boolean) =>
            changePermission(permission, enabled)
          }
        />
        <div className="flex flex-col space-y-4">
          <NormalButton
            onClick={() => mutation.mutate(editableRole)}
            disabled={mutation.isLoading}
            isLoading={mutation.isLoading}
            className="w-full md:w-20 hover:bg-gray-100 active:bg-gray-300"
          >
            Save
          </NormalButton>
          <DeleteRole id={role.id} />
        </div>
      </div>
    </div>
  )
}
