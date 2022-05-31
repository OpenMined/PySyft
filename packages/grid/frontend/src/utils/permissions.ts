import type { UserPermissions } from '@/types/grid-types'

export type GridPermissionsDescription = {
  [k in UserPermissions]: {
    name: string
    description: string
  }
}

export const gridPermissions: GridPermissionsDescription = {
  canTriageRequests: {
    name: 'Can triage requests',
    description: 'Allows users to accept or deny data requests',
  },
  canCreateGroups: {
    name: 'Can create groups',
    description: 'Allows user to create or edit groups',
  },
  canCreateUsers: {
    name: 'Can create users',
    description: 'Allows user to create or edit users',
  },
  canEditRoles: {
    name: 'Can edit roles',
    description: 'Allows user to create or edit roles',
  },
  canEditSettings: {
    name: 'Can edit settings',
    description: 'Allows user to manage Domain settings',
  },
  canManageInfrastructure: {
    name: 'Can manage infrastructure',
    description: 'Allows user to manage settings and control infrastructure',
  },
  canUploadData: {
    name: 'Can upload data',
    description: 'Allows user to upload new datasets',
  },
}
