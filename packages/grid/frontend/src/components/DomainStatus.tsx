import cn from 'classnames'
import { Text } from '@/omui'
import { useQuery } from 'react-query'
import api from '@/utils/api'
import { useEffect, useState } from 'react'

const statusColor = {
  online: 'bg-lime-500',
  offline: 'bg-red-500',
  loading: 'bg-gray-600',
}

const states = {
  online: 'domain online',
  offline: 'domain offline',
  loading: 'checking connection',
}

const ComponentStatus = ({
  noBox,
  status,
  size = 'sm',
}: {
  noBox: boolean
  status: keyof typeof states
  size?: string
}) => {
  return (
    <span className="items-center inline-flex capitalize space-x-2">
      {noBox ? (
        <CircleStatus status={status} />
      ) : (
        <div className="w-10 h-10 inline-flex justify-center items-center">
          <CircleStatus status={status} />
        </div>
      )}
      <Text size={size}>{states[status]}</Text>
    </span>
  )
}

const DomainStatus = ({
  noBox,
  textSize,
}: {
  noBox?: boolean
  textSize?: string
}) => {
  const [status, setStatus] = useState<keyof typeof states>('loading')
  const { isLoading, isError } = useQuery('domain-connection-status', () =>
    api.get('status').json()
  )

  useEffect(() => {
    if (isLoading) setStatus('loading')
    else if (isError) setStatus('offline')
    else setStatus('online')
  }, [isLoading, isError])

  return <ComponentStatus noBox={noBox} status={status} size={textSize} />
}

const CircleStatus = ({ status = 'loading', ...props }) => {
  return (
    <span
      aria-label={`Connection status: ${status}`}
      className={cn(
        'inline-block rounded-full w-1.5 h-1.5',
        statusColor[status],
        status === 'loading' && 'animate-pulse'
      )}
      {...props}
    />
  )
}

export { DomainStatus, CircleStatus }
