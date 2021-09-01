import {useEffect, useMemo, useState} from 'react'
import {useQueries} from 'react-query'
import {Page, NormalButton, SearchBar, SpinnerWithText, MutationError} from '@/components'
import {UserList, UserCreate} from '@/components/pages/users'
import {cacheKeys} from '@/utils'
import api from '@/utils/api-axios'
import {buildSelfWithRoles} from '@/lib/users/self'
import type {EnhancedUser, User, Me, Role} from '@/types/grid-types'
import type {ErrorMessage} from '@/utils/api-axios'

function buildUsersWithRoles(users: User[], roles: Role[]): EnhancedUser[] {
  return users.map(user => {
    const userRole = roles.find(role => role.name === user.role)
    const {id, name, ...permissions} = userRole
    return {
      permissions,
      user: {...user},
      role: {id, name}
    }
  })
}

type ShowUsers = {
  users: EnhancedUser[]
  me: Me
  isLoading: boolean
  isError: boolean
  error: string
}

function ShowUsers({users, me, isLoading, isError, error}: ShowUsers) {
  if (isLoading) {
    return <SpinnerWithText>Loading the user list</SpinnerWithText>
  }

  if (isError) {
    return <MutationError isError={isError} error="Unable to load the list of users" description={error} />
  }

  return <UserList users={users} me={me} />
}

interface SearchableUser {
  role: string
  email: string
}

function findMatchingUsers(users: EnhancedUser[], matches: SearchableUser[]) {
  return matches?.map(matchedUser => users.find(({user}) => user.email === matchedUser.email))
}

export default function Users() {
  const [showCreateUserPanel, setOpen] = useState<boolean>(false)
  const [users, setUsers] = useState<EnhancedUser[]>([])
  const [me, setMe] = useState<Me>(null)
  const [usersQuery, rolesQuery, meQuery] = useQueries([
    // TODO: Until grid endpoints are standardized, we do it "manually"
    {queryKey: cacheKeys.users, queryFn: () => api.get<{data: User[]}>(cacheKeys.users).then(res => res.data)},
    {queryKey: cacheKeys.roles, queryFn: () => api.get<{data: Role[]}>(cacheKeys.roles).then(res => res.data)},
    {queryKey: cacheKeys.me, queryFn: () => api.get<{data: User}>(cacheKeys.me).then(res => res.data)}
  ])

  const [filtered, setFiltered] = useState<EnhancedUser[]>(null)
  const searchFields = useMemo(() => ['email', 'role'], [])

  const isLoading = useMemo(
    () => Boolean(usersQuery.isLoading || rolesQuery.isLoading || meQuery.isLoading),
    [usersQuery.isLoading, rolesQuery.isLoading, meQuery.isLoading]
  )

  const isError = useMemo(
    () => Boolean(usersQuery.isError || rolesQuery.isError || meQuery.isError),
    [usersQuery.isError, rolesQuery.isError, meQuery.isError]
  )

  const error = useMemo(
    () => (usersQuery.error || rolesQuery.error || meQuery.error) as ErrorMessage,
    [usersQuery.error, rolesQuery.error, meQuery.error]
  )

  useEffect(() => {
    if (usersQuery.data && rolesQuery.data && meQuery.data) {
      const usersData = usersQuery.data as User[]
      const rolesData = rolesQuery.data as Role[]
      const meData = meQuery.data as User
      setUsers(buildUsersWithRoles(usersData, rolesData))
      setMe(buildSelfWithRoles(meData, rolesData))
    }
  }, [usersQuery.data, rolesQuery.data, meQuery.data])

  return (
    <Page title="Users" description="Manage users, edit user permissions and credentials">
      <section className="flex justify-between space-x-6">
        <SearchBar
          data={users.map(({user, role}): SearchableUser => ({role: role.name, email: user.email}))}
          searchFields={searchFields}
          setData={(matches: SearchableUser[]) => setFiltered(() => findMatchingUsers(users, matches))}
        />
        <NormalButton
          className={`flex-shrink-0 text-white bg-gray-900 bg-opacity-80 hover:bg-opacity-100`}
          onClick={() => setOpen(true)}>
          Create user
        </NormalButton>
      </section>
      {showCreateUserPanel && (
        <section>
          <UserCreate onClose={() => setOpen(false)} />
        </section>
      )}
      <section>
        <ShowUsers users={filtered ?? users} me={me} isLoading={isLoading} isError={isError} error={error?.message} />
      </section>
    </Page>
  )
}
