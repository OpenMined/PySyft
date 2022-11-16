import { useDomainStatus } from '@/lib/data'

export function DomainConnectionStatus() {
  const { isLoading, isError } = useDomainStatus()

  let bgColor = 'green'
  let message = 'Domain online'

  if (isError) {
    bgColor = 'red'
    message = 'Domain offline'
  }

  if (isLoading) {
    bgColor = 'gray'
    message = 'Checking connection...'
  }

  return (
    <div className="flex items-center p-4 space-x-2 text-sm">
      <div className="relative flex w-2 h-2">
        {isLoading && (
          <span className="absolute inline-flex w-full h-full bg-gray-400 rounded-full opacity-75 animate-ping"></span>
        )}
        <span className={`relative rounded-full w-2 h-2 bg-${bgColor}-500`} />
      </div>
      <p>{message}</p>
    </div>
  )
}
