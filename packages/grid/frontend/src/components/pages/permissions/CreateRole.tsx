import { useCallback, useState } from 'react'
import { useForm } from 'react-hook-form'
import { Input, NormalButton, PermissionsListToggle } from '@/components'
import { gridPermissions } from '@/utils'
import { useRoles } from '@/lib/data'
import type { Role, UserPermissions } from '@/types/grid-types'

const getBlankRole = () => ({
  ...(Object.keys(gridPermissions)
    .map((permission) => ({ permission }))
    .reduce((prev, curr) => {
      prev[curr.permission] = false
      return prev
    }, {}) as Record<UserPermissions, boolean>),
  name: '',
})

export function CreateRole({ onClose }: { onClose: () => void }) {
  const [newRole, setNewRole] = useState<Omit<Role, 'id'>>(getBlankRole)

  const { create } = useRoles()
  const mutation = create()

  const {
    register,
    handleSubmit,
    formState: { errors, isValid },
  } = useForm({ mode: 'onTouched' })

  const onSubmit = ({ name }) => {
    mutation.mutate({ ...newRole, name }, { onSuccess: onClose })
  }

  const changePermission = useCallback(
    (permission, enabled) =>
      setNewRole((role) => ({ ...role, [permission]: enabled })),
    []
  )

  return (
    <div className="p-8 space-y-6 rounded-md bg-blueGray-200">
      <header className="max-w-xl space-y-2">
        <h2 className="text-xl font-medium">Create a new role</h2>
        <p>Combine the permissions for creating a new role for your Domain.</p>
      </header>
      <form onSubmit={handleSubmit(onSubmit)}>
        <div className="space-y-4">
          <PermissionsListToggle
            onChange={(permission, enabled) =>
              changePermission(permission, enabled)
            }
            defaultPermissions={newRole}
            id="create-new-role"
          />
          <div className="max-w-md">
            <Input
              id="create-role-name"
              label="Role Name"
              name="name"
              ref={register}
              error={errors?.name}
              required
              placeholder="Role names are unique"
            />
          </div>
          <NormalButton
            className="flex-shrink-0 w-24 mt-auto mr-4 bg-gray-700 text-gray-50 bg-opacity-80 hover:bg-opacity-100"
            disabled={!isValid || mutation.isLoading}
            isLoading={mutation.isLoading}
          >
            Submit
          </NormalButton>
          <NormalButton
            type="button"
            className="flex-shrink-0 mt-auto"
            onClick={() => typeof onClose === 'function' && onClose()}
          >
            Close Panel
          </NormalButton>
        </div>
      </form>
    </div>
  )
}
