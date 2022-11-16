import { Fragment } from 'react'
import cn from 'classnames'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faChevronRight } from '@fortawesome/free-solid-svg-icons'
import { Text } from '@/omui'

function Breadcrumbs({ path }: { path: string }) {
  const split = path.split('.')

  return (
    <div className="flex p-2.5 text-gray-600 dark:text-gray-300 space-x-1">
      {split.map((crumb, index, original) => {
        const isLast = index + 1 === original.length
        const activeColors = 'text-primary-600 dark:text-primary-200'
        return (
          <Fragment key={`bc-${crumb}-${index}`}>
            {index > 0 && (
              <div
                className={cn(
                  isLast && activeColors,
                  'w-6 h-6 flex items-center justify-center text-xxs'
                )}
              >
                <FontAwesomeIcon icon={faChevronRight} />
              </div>
            )}
            <Text className={isLast && activeColors}>{crumb}</Text>
          </Fragment>
        )
      })}
    </div>
  )
}

export { Breadcrumbs }
