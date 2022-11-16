import { Text } from '@/omui'
import { usePermission } from './context'

function RoleTitle() {
  const { role } = usePermission()
  return (
    <Text bold size="lg">
      {role.name}
    </Text>
  )
}

// TODO: should come from the API
const defaultRoleIdAsString = '1' // default data scientist id

function DefaultRole() {
  const { role } = usePermission()

  if (String(role.id) !== defaultRoleIdAsString) return null

  return (
    <Text size="sm" className="italic">
      (default)
    </Text>
  )
}

function PermissionsAccordionTitle() {
  return (
    <div className="flex space-x-4 items-center">
      <RoleTitle />
      <DefaultRole />
    </div>
  )
}

export { RoleTitle, DefaultRole, PermissionsAccordionTitle }
