// TODO: should be shared between backend and frontend
export type SyftPermissions =
  | 'can_make_data_requests'
  | 'can_triage_data_requests'
  | 'can_manage_privacy_budgets'
  | 'can_create_users'
  | 'can_manage_users'
  | 'can_edit_roles'
  | 'can_upload_data'
  | 'can_upload_legal_documents'
  | 'can_edit_domain_settings'
  | 'can_manage_infrastructure'

export type AllSyftPermissions = {
  [k in SyftPermissions]: boolean
}

export type Role = {
  id: number
  name: string
} & AllSyftPermissions
