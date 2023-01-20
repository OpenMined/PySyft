import { faCheck, faUsers } from '@fortawesome/free-solid-svg-icons'
import { t } from '@/i18n'

export const sections = {
  users: {
    icon: faUsers,
    heading: t('users.heading', 'sections'),
    description: t('users.description', 'sections'),
  },
  permissions: {
    icon: faCheck,
    heading: t('permissions.heading', 'sections'),
    description: t('permissions.description', 'sections'),
  },
  settings: {
    icon: null,
    heading: t('settings.heading', 'sections'),
    description: t('settings.description', 'sections'),
  },
}
