import { useEffect, useState } from 'react'
import Link from 'next/link'
import { faCheck } from '@fortawesome/free-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Button, Divider, FormControl, Text } from '@/omui'
import { useRoles, useUsers } from '@/lib/data'
import { t } from '@/i18n'
import { allPermissions } from '@/components/Permissions/Panel'
import Modal from '../Modal'
import Select from '@/components/post/Select'

export function ChangeRoleModal({ show, onClose, user, role }) {
  const { data: roles } = useRoles().all()
  const { mutate } = useUsers().update(user?.id, { onSuccess: onClose })
  const [currentRole, setRole] = useState(role)
  const matchingRole = roles?.find((defRole) => defRole.name === currentRole)
  const permissionsList = allPermissions
    .filter((permission) => Boolean(matchingRole?.[permission]))
    .map((permission) => ({
      name: t(`${permission}.name`, 'permissions'),
      description: t(`${permission}.description`, 'permissions'),
    }))

  const handleRoleChange = () => {
    mutate({ role: currentRole })
  }

  useEffect(() => setRole(role), [role])

  if (!roles) return null

  return (
    <Modal show={show} onClose={onClose} className="max-w-3xl">
      <div className="col-span-full">
        <FontAwesomeIcon icon={faCheck} className="font-bold text-3xl" />
        <Text as="h1" className="mt-3" size="3xl">
          Change Roles
        </Text>
        <Text className="mt-4" as="p">
          Permissions for a user are set by their assigned role. These
          permissions are used for managing the domain. To review and customize
          the default set of roles visit the{' '}
          <Link href="/permissions">
            <a>Permissions</a>
          </Link>
          page.
        </Text>
      </div>
      <div className="col-span-full mt-2.5">
        <FormControl label="Change role" id="role" className="mt-6">
          <div>
            <Select
              options={roles}
              value={currentRole}
              onChange={(props) => setRole(props)}
            />
          </div>
        </FormControl>
      </div>
      <div className="col-span-full border border-gray-100 bg-gray-50 p-6 pb-8 space-y-4 mt-2.5 rounded">
        <Text>{t(`${currentRole}.description`, 'permissions')}</Text>
        <div className="py-4">
          <Divider color="light" />
        </div>
        <ul className="space-y-3">
          {permissionsList.map((permission) => (
            <li>
              <div className="flex space-x-6 items-center">
                <FontAwesomeIcon
                  icon={faCheck}
                  className="text-success-500 font-bold text-2xl"
                />
                <div className="py-1">
                  <Text
                    className="text-gray-800 capitalize"
                    bold
                    size="sm"
                    as="p"
                  >
                    {permission.name}
                  </Text>
                  <Text className="text-gray-400" size="sm" as="p">
                    {permission.description}
                  </Text>
                </div>
              </div>
            </li>
          ))}
        </ul>
      </div>
      <div className="col-span-full mt-6">
        <Link href="/permissions">
          <Button variant="link">Edit Role Permissions</Button>
        </Link>
      </div>
      <div className="col-span-full flex justify-between mt-12">
        <Button size="sm" variant="outline" onClick={onClose}>
          Cancel
        </Button>
        <Button size="sm" onClick={handleRoleChange}>
          Change Role
        </Button>
      </div>
    </Modal>
  )
}
