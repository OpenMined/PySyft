import {Accordion} from '@/components'
import {PermissionsAccordionTitle} from './PermissionsAccordionTitle'
import {PermissionsAccordionPanel} from '@/components/Permissions/Panel'
import {PermissionsAccordionProvider} from './context'
import type {Role} from '@/types/permissions'

export function PermissionsAccordion({roles}: {roles: Array<Role>}) {
  return (
    <Accordion>
      {roles.map(role => (
        <PermissionsAccordionProvider role={role}>
          <Accordion.Item key={role.name} openClasses="bg-gray-50">
            <PermissionsAccordionTitle />
            <PermissionsAccordionPanel />
          </Accordion.Item>
        </PermissionsAccordionProvider>
      ))}
    </Accordion>
  )
}
