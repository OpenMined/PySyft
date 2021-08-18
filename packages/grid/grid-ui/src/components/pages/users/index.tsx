import {createContext, useContext, useState} from 'react'
import Link from 'next/link'
import {useQueryClient, useMutation, useQuery} from 'react-query'
import {useForm} from 'react-hook-form'
import {XIcon, CheckIcon} from '@heroicons/react/outline'
import {NormalButton, DeleteButton, Input, Select, Badge, List, AccordionListItem} from '@/components'
import {entityColors, gridPermissions, cacheKeys} from '@/utils'
import api from '@/utils/api-axios'
import type {ChangeEvent} from 'react'
import type {EnhancedUser, User, Role, Me} from '@/types/grid-types'

interface UsersAsProp {
  users: EnhancedUser[]
}

const UserListContext = createContext(null)

export function UserList({users, me}: UsersAsProp & {me: Me}) {
  return (
    <List>
      {users.map(({user, role, permissions}: EnhancedUser) => (
        <UserListContext.Provider value={{user, role, permissions, me}} key={`list-${user.email}`}>
          <AccordionListItem>
            <UserInfoTitle />
            <UserInfoPanel />
          </AccordionListItem>
        </UserListContext.Provider>
      ))}
    </List>
  )
}

function UserInfoTitle() {
  const {user, role} = useContext(UserListContext)
  return (
    <div>
      <div className="flex space-x-2 truncate">
        <p className="font-medium truncate">{user.name}</p>
        <Badge bgColor={entityColors.roles}>{role.name}</Badge>
      </div>
      <p className="text-sm text-gray-500">{user.email}</p>
    </div>
  )
}

function ShowPermissions() {
  return (
    <div className="max-w-xl">
      <h3 className="text-base font-medium">User permissions</h3>
      <div className="space-y-2">
        <p>Permissions are set by roles and they are used for managing the domain.</p>
        <p>
          For example, a Data Scientist is not expected to have any managerial permissions such as editing users or
          uploading datasets, while a Data Compliance Office should have at least the permission to triage requests. A
          Data Owner has all permissions.
        </p>
        <p>
          If you want to modify the permissions, you can change the user role or{' '}
          <Link href="/permissions">
            <a className="underline">reorganize the permissions</a>
          </Link>
          .
        </p>
        <PermissionsList />
      </div>
    </div>
  )
}

function PermissionsList() {
  const {permissions} = useContext(UserListContext)
  return (
    <div className="mx-2 my-4 space-y-1">
      {Object.keys(permissions).map(permission => (
        <div key={permission}>
          <div className="flex items-center space-x-2">
            {permissions[permission] ? (
              <>
                <CheckIcon className="w-4 h-4 mr-2" />
                <p className="italic font-medium">{gridPermissions[permission]?.name}</p>
              </>
            ) : (
              <>
                <XIcon className="w-3 h-3 mr-2 text-red-700" />
                <p>{gridPermissions[permission]?.name}</p>
              </>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

function ChangeRole() {
  const {user, role: userRole} = useContext(UserListContext)
  const [newRole, chooseNewRole] = useState<string>(String(userRole.id))
  const queryClient = useQueryClient()
  const invalidate = () => queryClient.invalidateQueries(cacheKeys.users)
  const {data: allRoles} = useQuery<Role[]>(cacheKeys.roles)
  const mutation = useMutation(() => api.patch(`${cacheKeys.users}/${user.id}`, {role: newRole}), {
    onSuccess: invalidate
  })

  return (
    <div className="flex max-w-xl space-x-4">
      <Select
        id="user-roles"
        label="Change user role"
        container="flex-grow w-full"
        options={allRoles.map(role => ({value: String(role.name), label: role.name}))}
        className="overflow-hidden truncate"
        value={newRole}
        onChange={e => chooseNewRole(e.target.value)}
      />
      <NormalButton
        onClick={() => mutation.mutate()}
        className="flex-shrink-0 w-24 mt-auto"
        disabled={mutation.isLoading}
        isLoading={mutation.isLoading}>
        Submit
      </NormalButton>
    </div>
  )
}

function ChangePassword() {
  const {user} = useContext(UserListContext)
  const [password, setPassword] = useState<string>('')
  const queryClient = useQueryClient()
  const invalidate = () => queryClient.invalidateQueries(cacheKeys.users)
  const mutation = useMutation(() => api.patch(`${cacheKeys.users}/${user.id}`, {password}), {
    onSuccess: invalidate
  })

  return (
    <div className="flex max-w-xl space-x-4">
      <Input
        id={`user-password-${user.id}`}
        type="password"
        placeholder="This overrides the user password"
        label="Change user password"
        container="flex-grow w-full"
        onChange={e => setPassword(e.target.value)}
        value={password}
      />
      <NormalButton
        className="flex-shrink-0 w-24 mt-auto"
        disabled={mutation.isLoading}
        onClick={() => mutation.mutate()}
        isLoading={mutation.isLoading}>
        Submit
      </NormalButton>
    </div>
  )
}

function ChangeEmail() {
  const {user} = useContext(UserListContext)
  const [email, setEmail] = useState<string>(user.email)
  const queryClient = useQueryClient()
  const invalidate = () => queryClient.invalidateQueries(cacheKeys.users)
  const mutation = useMutation(() => api.patch(`${cacheKeys.users}/${user.id}`, {email}), {onSuccess: invalidate})

  return (
    <div className="flex max-w-xl space-x-4">
      <Input
        id={`user-email-${user.email}`}
        label="Change user email"
        container="flex-grow w-full"
        onChange={(e: ChangeEvent<HTMLInputElement>) => setEmail(e.target.value)}
        value={email}
      />
      <NormalButton
        className="flex-shrink-0 w-24 mt-auto"
        disabled={mutation.isLoading}
        onClick={() => mutation.mutate()}
        isLoading={mutation.isLoading}>
        Submit
      </NormalButton>
    </div>
  )
}

function useInvalidate(queries: string | string[]) {
  const queryClient = useQueryClient()
  return () => queryClient.invalidateQueries(queries)
}

function DeleteUser() {
  const {user} = useContext(UserListContext)
  const invalidate = useInvalidate(cacheKeys.users)
  const mutation = useMutation(() => api.delete(`${cacheKeys.users}/${user.id}`), {onSuccess: invalidate})

  return (
    <div>
      <DeleteButton
        className="w-32"
        isLoading={mutation.isLoading}
        disabled={mutation.isLoading}
        onClick={() => mutation.mutate()}>
        Delete User
      </DeleteButton>
    </div>
  )
}

function UserInfoPanel() {
  const {user, permissions, me} = useContext(UserListContext)
  return (
    <div className="py-6 pl-16 pr-4 space-y-6 text-sm border-t border-gray-100 bg-blueGray-100">
      <ShowPermissions {...permissions} />
      {me.permissions.canEditRoles && <ChangeRole />}
      {(user.id === me.id || me.permissions.canCreateUsers) && (
        <>
          <ChangeEmail />
          <ChangePassword />
        </>
      )}
      {user.id !== 1 && me.permissions.canCreateUsers && <DeleteUser />}
    </div>
  )
}

interface UserSignUp {
  name: string
  email: string
  password: string
  role: string
}

export function UserCreate({onClose}: {onClose: () => void}) {
  const {
    register,
    handleSubmit,
    reset,
    formState: {errors, isValid}
  } = useForm({mode: 'onTouched'})
  const {data: allRoles} = useQuery<Role[]>(cacheKeys.roles)
  const options = allRoles?.map(role => ({value: String(role.name), label: role.name}))
  const queryClient = useQueryClient()
  const invalidate = () => queryClient.invalidateQueries([cacheKeys.users])

  const mutation = useMutation((user: UserSignUp) => api.post<User>(cacheKeys.users, user), {
    onSuccess: () => {
      invalidate()
      reset()
      typeof onClose === 'function' && onClose()
    }
  })

  const onSubmit = (values: UserSignUp) => {
    mutation.mutate(values)
  }

  return (
    <div className="p-8 space-y-6 rounded-md bg-blueGray-200">
      <header className="max-w-xl space-y-2">
        <h2 className="text-xl font-medium">Create a new users</h2>
        <p>
          PyGrid utilizes users and roles to appropriately permission data at a higher level. All users with the
          permission{' '}
          <span className="p-1 text-xs uppercase bg-gray-100 text-trueGray-800 tracker-tighter">Can create users</span>{' '}
          are allowed to create new users in the domain.
        </p>
      </header>
      <form onSubmit={handleSubmit(onSubmit)}>
        <div className="max-w-md space-y-4">
          <Input id="create-user-name" label="Full name" name="name" ref={register} error={errors.name} required />
          <Input id="create-user-email" label="User email" name="email" ref={register} error={errors.email} required />
          <Input
            id="create-user-password"
            type="password"
            label="User Password"
            name="password"
            ref={register}
            error={errors.password}
            required
          />
          <Select
            id="create-user-roles"
            label="Change user role"
            name="role"
            placeholder="Select a role"
            options={options}
            className="overflow-hidden truncate"
            error={errors.role}
            ref={register}
            required
          />
          <NormalButton
            className="flex-shrink-0 w-24 mt-auto mr-4 bg-gray-700 text-gray-50 bg-opacity-80 hover:bg-opacity-100"
            disabled={!isValid || mutation.isLoading}
            isLoading={mutation.isLoading}>
            Submit
          </NormalButton>
          <NormalButton
            type="button"
            className="flex-shrink-0 mt-auto"
            onClick={() => typeof onClose === 'function' && onClose()}>
            Close Panel
          </NormalButton>
        </div>
      </form>
    </div>
  )
}
