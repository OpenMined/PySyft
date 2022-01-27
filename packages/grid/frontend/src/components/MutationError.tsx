import { Alert } from '@/components'

const GENERIC_ERROR = 'There was an error'
const GENERIC_ERROR_DESCRIPTION = 'Check your connection status'

export function MutationError({ isError, error, description }) {
  if (!isError) {
    return null
  }

  return (
    <Alert error={error ?? GENERIC_ERROR} description={description ?? GENERIC_ERROR_DESCRIPTION} />
  )
}
