export interface Network {
  joined_on?: Date | string
  status?: string
  id: string
  description?: string
  url: string
  tags?: Array<string>
  total_domains?: number
  total_datasets?: number
}
