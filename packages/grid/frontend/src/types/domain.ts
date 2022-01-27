import type { Network } from './network.ts'

export interface Domain {
  id: string
  name: string
  description?: string
  email: string
  total_datasets: number
  created_on: Date | string
  owner: string
  company?: string
  networks: Array<Partial<Network>>
  tags: Array<string>
  version: string
}
