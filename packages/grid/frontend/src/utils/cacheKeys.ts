const DATASETS = 'datasets'
const SETUP = 'settings'
const USERS = 'users'
const ROLES = 'roles'
const ASSOCIATION_REQUEST = 'association-requests'
const STATUS = 'status'
const REQUESTS = 'requests'
const APPLICANT = 'applicants'

export const cacheKeys = {
  association_request: `${ASSOCIATION_REQUEST}`,
  applicant_users: `${USERS}/${APPLICANT}`,
  requests: `${REQUESTS}`,
  data: `${REQUESTS}/data`,
  budget: `${REQUESTS}/budget`,
  datasets: `${DATASETS}`,
  users: `${USERS}`,
  me: `${USERS}/me`,
  roles: `${ROLES}`,
  settings: `${SETUP}`,
  status: `${STATUS}`,
  ping: `ping`,
}
