import { Toggle } from '@/components'
import { gridPermissions } from '@/utils'
import type { UserPermissions, Role } from '@/types/grid-types'

interface PermissionListWithToggle {
  onChange: (e: UserPermissions, e2: boolean) => void
  defaultPermissions: Omit<Role, 'id'>
  id: string | number
}

export function PermissionsListToggle({
  onChange,
  defaultPermissions,
  id,
}: PermissionListWithToggle) {
  return (
    <>
      {Object.keys(gridPermissions).map((permission: UserPermissions) => (
        <div key={`${id}-${permission}`} className="text-sm">
          <div className="flex items-center justify-between space-x-2">
            <p className="font-medium">{gridPermissions[permission]?.name}</p>
            <div className="flex-shrink-0">
              <Toggle
                alt={`Allow ${permission}`}
                initialEnabled={defaultPermissions?.[permission]}
                onChange={(enabled: boolean) => onChange(permission, enabled)}
              />
            </div>
          </div>
          <p className="text-gray-500">
            {gridPermissions[permission]?.description}
          </p>
        </div>
      ))}
    </>
  )
}
