import { Badge } from '@/omui'

export function RequestStatusBadge({ status }) {
  if (status === 'pending')
    return (
      <Badge variant="primary" type="solid">
        Pending
      </Badge>
    )
  if (status === 'accepted')
    return (
      <Badge variant="success" type="solid">
        Accepted
      </Badge>
    )
  if (status === 'denied')
    return (
      <Badge variant="danger" type="solid">
        Rejected
      </Badge>
    )
  return (
    <Badge variant="gray" type="subtle" className="capitalize">
      {status}
    </Badge>
  )
}
