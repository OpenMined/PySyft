import { Text } from '@/omui'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faSpinner } from '@fortawesome/free-solid-svg-icons'
import { t } from '@/i18n'

function Loader({ type = 'grid' }) {
  return (
    <div className="flex items-center space-x-2">
      {type === 'grid' && (
        <img
          src="/assets/small-grid-symbol-logo.png"
          width="32"
          className="animate-spin"
        />
      )}
      {type === 'spinner' && (
        <FontAwesomeIcon icon={faSpinner} className="animate-spin text-3xl" />
      )}
      <Text className="text-gray-500">{t('loading')}</Text>
    </div>
  )
}

export { Loader }
