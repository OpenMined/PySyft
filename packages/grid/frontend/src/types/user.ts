import type { Role } from './role'

export interface User {
  id: string
  name: string
  role: Role
  email: string
  budget_spent: number
  budget: number
  created_on: Date | string
  added_by: {
    id: string
    name: string
    role: Role
  }
  institution: string
  website: string
}
