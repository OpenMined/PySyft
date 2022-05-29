import { usePermission } from './context'
import { Button, Divider, Switch, Text } from '@/omui'
import { t } from '@/i18n'

import type { SyftPermissions } from '@/types/permissions'

// TODO: evaluate creating API endpoints for serving each permission type (!== roles)
// TODO: should be created via a shared object between backend and frontend
// prettier-ignore
export const allPermissions: Array<SyftPermissions> = [
  'can_make_data_requests', 'can_edit_roles',
  'can_triage_data_requests', 'can_upload_data',
  'can_manage_privacy_budgets', 'can_upload_legal_documents',
  'can_manage_users', 'can_edit_domain_settings',
  'can_create_users', 'can_manage_infrastructure'
]

// TODO: Remove? Move to API endpoint?
const disabledPermissions = ['can_manage_infrastructure']

function RoleDescription() {
  const { role } = usePermission()
  return <Text as="p">{t(`${role.name}.description`, 'permissions')}</Text>
}

function PermissionToggler({ permission }: { permission: SyftPermissions }) {
  const { permissions, toggle } = usePermission()
  const isChecked = permissions[permission]
  const isDisabled = disabledPermissions.includes(permission)

  return (
    <div className="flex items-center space-x-6">
      <Switch
        disabled={isDisabled}
        checked={isChecked}
        onChange={() => toggle(permission)}
      />
      <div className="w-full flex flex-col">
        <Text size="sm" bold className="capitalize">
          {t(`${permission}.name`, 'permissions')}
        </Text>
        <Text size="sm" className="text-gray-400">
          {t(`${permission}.description`, 'permissions')}
        </Text>
      </div>
    </div>
  )
}

function PermissionsAccordionPanel() {
  const { save, isLoading } = usePermission()

  return (
    <div className="px-10 w-full space-y-6">
      <RoleDescription />
      <Divider color="light" />
      {/* change to flex box, grid da padding errado */}
      <div className="grid grid-cols-2 gap-y-4 gap-x-6">
        {allPermissions.map((permission) => (
          <div key={permission} className="flex px-2 items-start">
            <PermissionToggler permission={permission} />
          </div>
        ))}
      </div>
      <Button
        variant="primary"
        type="button"
        onClick={save}
        isLoading={isLoading}
      >
        {t('buttons.save-changes')}
      </Button>
    </div>
  )
}

export { PermissionsAccordionPanel }
