import type { SyftUID } from "./syft"

export interface UserRole {
  value: ServiceRoles
}

export interface UserListView {
  id: SyftUID
  role: UserRole
  name: string
  email: string
}

export interface UserView {
  id: SyftUID
  role: UserRole
  name: string
  email: string
  institution?: string
  website?: string
}

export enum ServiceRoles {
  NONE = 0,
  GUEST = 1,
  DATA_SCIENTIST = 2,
  DATA_OWNER = 32,
  ADMIN = 128,
}
