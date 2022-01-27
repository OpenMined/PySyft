import { Badge } from '$components/Badge'
import type { BadgeProps } from '$components/Badge'

type IdProps = { id: string } & BadgeProps

export const Id = ({ id, ...badgeProps }: IdProps) => <Badge {...badgeProps}>{id}</Badge>
