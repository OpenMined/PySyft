import { Badge } from '@/omui'

function RoleBadge({ role }: { role: string }) {
  return (
    <Badge variant="primary" type="subtle" className="self-center">
      {role}
    </Badge>
  )
}

export { RoleBadge }
