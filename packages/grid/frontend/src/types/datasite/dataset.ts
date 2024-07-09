import type { SyftUID } from "./syft"

export interface DatasetContributor {
  id: SyftUID
  email: string
  name: string
  note?: string
  phone?: string
  role?: string
}

export interface DatasetDataSubject {
  id: SyftUID
  server_uid: SyftUID
  aliases: string[]
  description?: string
  name: string
}

export interface DatasetAsset {
  id: SyftUID
  action_id: SyftUID
  server_uid: SyftUID
  contributors: DatasetContributor[]
  data_subjects: DatasetDataSubject[]
  description: string
  mock_is_real: boolean
  name: string
  shape: number[]
}

export interface Dataset {
  id: SyftUID
  server_id: SyftUID
  asset_list: DatasetAsset[]
  contributors: DatasetContributor[]
  citation: string
  name: string
  description: string
  requests: number
  mb_size: number
  updated_at: string
  url: string
}
