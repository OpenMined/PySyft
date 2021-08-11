import {useRoles, useMe} from '@/lib/data'
import type {User, Role, Me} from '@/types/grid-types'

export function buildSelfWithRoles(me: User, roles: Role[]): Me {
  const selfRole = roles.find(role => role.name === me.role)
  const {id: selfRoleId, name: selfRoleName, ...selfPermissions} = selfRole
  return {
    id: me.id,
    email: me.email,
    permissions: selfPermissions,
    role: {id: selfRoleId, name: selfRoleName}
  }
}

export function useEnhancedCurrentUser() {
  const {all: getAllRoles} = useRoles()
  const {data: roles} = getAllRoles()
  const {data: me} = useMe()
  if (!roles || !me) {
    return null
  }
  return buildSelfWithRoles(me, roles)
}
