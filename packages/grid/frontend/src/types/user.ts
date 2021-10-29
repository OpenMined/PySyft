import type {Role} from './role'

export interface User {
  id: string
  name: string
  role: Role
  email: string
  current_balance: number
  allocated_budget: number
  created_on: Date | string
  added_by: {
    id: string
    name: string
    role: Role
  }
  institution: string
  website: string
}
