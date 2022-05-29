import { faCheck, faSpinner } from '@fortawesome/free-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { H4, ListInnerContainer, Select, Text } from '@/omui'
import { SingleCenter } from '@/components/Layouts'
import { PermissionsAccordion } from '@/components/Permissions/PermissionAccordion'
import { TopContent } from '@/components/lib'
import { useRoles } from '@/lib/data'
import { singularOrPlural } from '@/utils'
import { t } from '@/i18n'
import { sections } from '@/content'
import { NewSpinner } from '@/components/NewSpinner'

export default function Permissions() {
  const { data: roles, isLoading } = useRoles().all()
  const results = { length: 4 }

  return (
    <SingleCenter>
      <TopContent
        heading={sections.permissions.heading}
        icon={() => <FontAwesomeIcon icon={faCheck} className="text-3xl" />}
      />
      <Text as="p" className="col-span-7 mt-8 text-gray-600">
        {sections.permissions.description}
      </Text>
      <div className="mt-12 col-span-4 flex items-center">
        <button className="w-full">
          <Select placeholder="Filter by permissions" />
        </button>
        <ListInnerContainer>
          <div className="rounded-full bg-gray-200 w-1.5 h-1.5" />
        </ListInnerContainer>
        <Text className="flex-shrink-0">
          {results.length}{' '}
          {singularOrPlural(t('result'), t('results'), results.length)}
        </Text>
      </div>
      <div className="col-span-full mt-10 space-y-4">
        <H4>Roles</H4>
        {isLoading && <NewSpinner className="text-blue-500 w-6 h-6" />}
        {roles?.length > 0 && <PermissionsAccordion roles={roles} />}
      </div>
    </SingleCenter>
  )
}
