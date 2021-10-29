import {useQuery} from 'react-query'
import {cacheKeys} from '@/utils'

export function DomainConnectionStatus() {
  const {isLoading, isError} = useQuery(cacheKeys.status)

  if (isError) {
    return (
      <div className="flex">
        <div className="rounded-full w-2 h-2 bg-red-500" />
        <p>Domain offline</p>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex">
        <div className="animate-pulse rounded-full w-2 h-2 bg-gray-100" />
        <p>Checking connection...</p>
      </div>
    )
  }

  return (
    <div className="flex">
      <div className="rounded-full w-2 h-2 bg-green-500" />
      <p>Domain reachable</p>
    </div>
  )
}
