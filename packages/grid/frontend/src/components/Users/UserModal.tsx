import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faInfoCircle, faTrash } from '@fortawesome/free-solid-svg-icons'
import { Button, Divider, H2, H6, Text } from '@/omui'
import { RoleBadge } from '@/components/RoleBadge'
import Modal from '../Modal'
import { t } from '@/i18n'
import { formatBudget, formatDate } from '@/utils'
import { BorderedBox } from '@/components/Boxes'
import { useUsers } from '@/lib/data'

function UserModal({ show, onClose, user, onEditRole, onAdjustBudget }) {
  const removeUser = useUsers().remove(user?.id, { onSuccess: onClose }).mutate

  if (!user) return null

  return (
    <Modal show={show} onClose={onClose} withExpand={`/active/${user.id}`}>
      <div className="col-span-10 col-start-2 mt-10">
        <div className="flex justify-between">
          <div className="flex space-x-4">
            <H2 className="items-center">{user.name}</H2>
            <RoleBadge role={user.role} />
          </div>
          <Button variant="ghost" size="sm" className="flex-shrink-0" onClick={removeUser}>
            <Text size="sm" className="text-gray-400">
              <FontAwesomeIcon icon={faTrash} className="mr-2" /> {t('delete-user')}
            </Text>
          </Button>
        </div>
        <button onClick={onEditRole}>
          <Text size="sm" as="p" className="text-primary-600 mt-3" underline>
            {t('change-role')}
          </Text>
        </button>
      </div>
      <div className="grid grid-cols-10 col-span-10 col-start-2 gap-8 mt-8 mb-10">
        <PrivacyBudgetAdjustCard {...user} onAdjustBudget={onAdjustBudget} />
        <Background {...user} />
        <System {...user} />
      </div>
    </Modal>
  )
}

function PrivacyBudgetAdjustCard({ budget_spent, budget, onAdjustBudget }) {
  return (
    <div className="col-span-7 space-y-3">
      <H6 bold>
        {t('privacy-budget')} <FontAwesomeIcon icon={faInfoCircle} />
      </H6>
      <div className="w-full flex items-center p-6 border-gray-100 border rounded bg-gray-50 space-x-6 justify-between">
        <div className="flex pb-3 space-x-4">
          <div>
            <Text as="p" size="lg" bold className="text-error-600">
              {formatBudget(budget_spent)} ɛ
            </Text>
            <Text as="p" className="capitalize">
              {t('current-balance')}
            </Text>
          </div>
          <Divider orientation="vertical" color="light" className="self-stretch py-4" />
          <div>
            <Text as="p" bold size="lg">
              {formatBudget(budget)} ɛ
            </Text>
            <Text as="p" className="capitalize">
              {t('allocated-budget')}
            </Text>
          </div>
        </div>
        <Button size="sm" variant="outline" onClick={onAdjustBudget}>
          {t('buttons.adjust-budget')}
        </Button>
      </div>
    </div>
  )
}

function Background({ email, institution, website }) {
  const info = [
    { text: t('email'), value: email, link: Boolean(email) },
    { text: t('company-institution'), value: institution },
    { text: t('website-profile'), value: website, link: Boolean(website) },
  ]
  return (
    <div className="col-span-full space-y-3">
      <H6 bold>{t('background')}</H6>
      <BorderedBox className="space-y-4">
        {info.map(uinfo => (
          <Text key={uinfo.text} as="p" size="sm" bold>
            {uinfo.text}:
            <Text size="sm" underline={uinfo.link} className="ml-2">
              {uinfo.value || '--'}
            </Text>
          </Text>
        ))}
      </BorderedBox>
    </div>
  )
}

function System({ created_at, added_by, daa_pdf, daa_pdf_uploaded_on }) {
  const info = [
    { text: t('date-added'), value: formatDate(created_at) },
    { text: t('added-by'), value: added_by },
    { text: t('data-access-agreement'), value: daa_pdf },
    { text: t('uploaded-on'), value: daa_pdf_uploaded_on },
  ]

  return (
    <div className="col-span-full space-y-3">
      <H6 bold>{t('system')}</H6>
      <BorderedBox className="space-y-4">
        {info.map(uinfo => (
          <Text key={uinfo.text} as="p" size="sm" bold>
            {uinfo.text}
            <Text size="sm" className="ml-2">
              {uinfo.value ?? '--'}
            </Text>
          </Text>
        ))}
      </BorderedBox>
    </div>
  )
}

export { UserModal }
