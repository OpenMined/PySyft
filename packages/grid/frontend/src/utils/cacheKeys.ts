export const DATA_CENTRIC = ''
export const SETUP = '/setup'
export const USERS = '/users'
export const ROLES = '/roles'

export const cacheKeys = {
  requests: `${DATA_CENTRIC}/requests`,
  models: `${DATA_CENTRIC}/models`,
  tensors: `${DATA_CENTRIC}/tensors`,
  datasets: `${DATA_CENTRIC}/datasets`,
  users: `${USERS}`,
  me: `${USERS}/me`,
  roles: `${ROLES}`,
  settings: `${SETUP}`,
  status: `/status`
}
