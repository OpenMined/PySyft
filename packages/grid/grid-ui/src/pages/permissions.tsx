import {useMemo, useState} from 'react'
import {flatten, uniq} from 'lodash'
import {Page, NormalButton, SpinnerWithText, MutationError} from '@/components'
import {PermissionList, CreateRole} from '@/components/pages/permissions'
import {PermissionsFilter} from '@/components/FilterMenu'
import {useRoles} from '@/lib/data'
import {useEnhancedCurrentUser} from '@/lib/users/self'
import {gridPermissions} from '@/utils'
import type {Role} from '@/types/grid-types'

const permissionList = Object.keys(gridPermissions)

interface PermissionsList {
  roles: Role[]
  isLoading: boolean
  isError: boolean
  errorMessage: string
}

function PermissionsList({roles, isLoading, isError, errorMessage}: PermissionsList) {
  if (isLoading) return <SpinnerWithText>Loading the list of available tensors</SpinnerWithText>
  if (isError) return <MutationError isError error="Unable to load tensors" description={errorMessage} />
  return <PermissionList roles={roles} />
}

export default function Permissions() {
  const {all} = useRoles()
  const {data: roles, isLoading, isError, error} = all()
  const me = useEnhancedCurrentUser()

  const [openCreatePanel, setOpen] = useState<boolean>(false)

  const [filters, setFilters] = useState([])
  const filtered = useMemo(
    () =>
      (filters.some(filter => filter)
        ? uniq(
            flatten(
              filters.map((isSelected, index) => {
                if (isSelected) {
                  const permission = permissionList[index]
                  return roles.filter(role => role[permission])
                }
                return null
              })
            ).filter((role: Role) => role)
          )
        : roles ?? []
      ).sort((a: Role, b: Role) => a.id - b.id),
    [filters, roles]
  )

  return (
    <Page title="Roles and Permissions" description="Create new roles and edit the permissions attached to each role">
      <section className="flex justify-between">
        <PermissionsFilter setSelectedFilters={setFilters} />
        {me?.permissions?.canEditRoles && (
          <div>
            <NormalButton
              className={`flex-shrink-0 bg-gray-900 text-white bg-opacity-80 hover:bg-opacity-100`}
              onClick={() => setOpen(true)}>
              Create Role
            </NormalButton>
          </div>
        )}
      </section>
      {openCreatePanel && (
        <section>
          <CreateRole onClose={() => setOpen(false)} />
        </section>
      )}

      <section>
        <PermissionsList roles={filtered} isLoading={isLoading} isError={isError} errorMessage={error?.message} />
      </section>
    </Page>
  )
}
