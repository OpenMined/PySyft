export interface Tensor {
  name?: string
  id: string
  dtype?: 'Tensor'
  shape?: string
  description?: string | string[]
  tags?: string[]
}

export interface Dataset {
  createdAt: Date | string
  data: Tensor[]
  description: string | string[]
  id: string
  manifest: string | string[]
  name: string
  tags: string[]
}

export interface Model {
  name: string
  tags: string[]
  description: string | string[]
  id: string
  createdAt?: Date | string
}

export type RequestStatus = 'pending' | 'accepted' | 'denied'

export interface Request {
  id: string
  date: Date | string
  userId: string | number
  userName: string
  objectId: string
  reason: string | string[]
  status: RequestStatus
  request_type: 'permissions'
  verifyKey: string
  objectType: string
  tags: string[]
}

export interface User {
  name: string
  email: string
  id: number
  role: string
}

export interface UserMe extends Omit<User, 'role'> {
  role: number
}

export type UserPermissions =
  | 'canTriageRequests'
  | 'canEditSettings'
  | 'canCreateUsers'
  | 'canCreateGroups'
  | 'canEditRoles'
  | 'canManageInfrastructure'
  | 'canUploadData'

export type GridPermissions = {
  [k in UserPermissions]: boolean
}

export type Role = {
  id: number
  name: string
} & GridPermissions

export interface EnhancedUser {
  user: Partial<User>
  permissions: GridPermissions
  role: Partial<Role>
}

export interface Me {
  id: number
  email: string
  permissions: GridPermissions
  role: Pick<Role, 'id' | 'name'>
}

export interface Settings {
  // FIX: Which one is it? :)
  domainName: string
  nodeName: string
  id: number
}

export interface DomainStatus {
  nodeName: string
  init: boolean
}
