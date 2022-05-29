import { useMemo, useState } from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { useForm, Controller } from 'react-hook-form'
import {
  faCalendar,
  faTrash,
  faUser,
  faEnvelope,
  faPlus,
  faUserPlus,
} from '@fortawesome/free-solid-svg-icons'
import { useDisclosure } from '@/hooks/useDisclosure'
import { Badge, Button, Divider, Input, H4, Select, Tabs, Text } from '@/omui'
import { NameAndBadge, SearchInput, TopContent, Dot } from '@/components/lib'
import { Alert } from '@/components/Alert'
import { Base } from '@/components/Layouts'
import { DeleteModal } from '@/components/Users/DeleteModal'
import { AcceptDeny } from '@/components/AcceptDenyButtons'
import Modal from '@/components/Modal'
import { TableItem, useOMUITable } from '@/components/Table'
import { useApplicantUsers, useRoles, useUsers } from '@/lib/data'
import { sections } from '@/content'
import { formatDate } from '@/utils'
import { UserModal } from '@/components/Users/UserModal'

import commonStrings from '@/i18n/en/common.json'
import usersStrings from '@/i18n/en/users.json'
import { ChangeRoleModal } from '@/components/Users/ChangeRoleModal'
import { PrivacyBudgetModal } from '@/components/Users/PrivacyBudgetModal'

function Active() {
  const { data: roles } = useRoles().all()
  const { data: users } = useUsers().all()
  const [isCreatingUser, showCreateUser] = useState(false)
  const [selectedUser, setSelectedUser] = useState(null)
  const [selectedModal, setModal] = useState(null)

  return (
    <>
      <div className="col-span-11 mt-10">
        <Alert.Info
          alertStyle="topAccent"
          description={usersStrings.active.alert}
        />
      </div>
      {/* <div className="flex col-span-5 items-center mt-10"> */}
      {/*   <SearchInput /> */}
      {/*   <Dot /> */}
      {/*   <div className="flex-shrink-0 h-full"> */}
      {/*     <Select placeholder="Role" options={roles?.map?.(role => ({label: role.name, value: role.id}))} /> */}
      {/*   </div> */}
      {/* </div> */}
      {/* <div className="col-span-3 col-start-10 text-right mt-10"> */}
      <div className="col-span-full mt-10">
        <Button variant="gray" onClick={() => showCreateUser(true)}>
          <Text bold size="sm">
            <FontAwesomeIcon icon={faPlus} className="mr-2" /> Create User
          </Text>
        </Button>
      </div>
      <Divider color="light" className="col-span-full mt-8" />
      <div className="col-span-full mt-4">
        <ActiveUsersTable
          users={users}
          setSelectedUser={(user) => {
            setSelectedUser(user)
            setModal('user')
          }}
        />
      </div>
      <Modal show={isCreatingUser} onClose={() => showCreateUser(false)}>
        <CreateUser onClose={() => showCreateUser(false)} />
      </Modal>
      {selectedUser && (
        <UserModal
          show={selectedModal === 'user'}
          user={users.find((user) => user.id === selectedUser.id)}
          onClose={() => setModal('')}
          onEditRole={() => setModal('change-role')}
          onAdjustBudget={() => setModal('adjust-budget')}
        />
      )}
      {selectedUser && (
        <ChangeRoleModal
          show={selectedModal === 'change-role'}
          onClose={() => setModal('user')}
          role={selectedUser?.role}
          user={selectedUser}
        />
      )}
      {selectedUser && (
        <PrivacyBudgetModal
          show={selectedModal === 'adjust-budget'}
          onClose={() => setModal('user')}
          user={selectedUser}
        />
      )}
    </>
  )
}

function DeniedUsersTable({ users }) {
  const tableData = useMemo(
    () =>
      users?.map((user) => ({
        ...user,
        summary: {
          name: user.name,
          id: user.id,
          role: user.role,
        },
      })) ?? [],
    [users]
  )

  const tableColumns = useMemo(
    () => [
      {
        Header: 'Name',
        accessor: (d) => d.name,
        id: 'tab_name',
        Cell: ({ cell: { value } }) => <Text size="sm">{value}</Text>,
      },
      {
        Header: (
          <Text size="sm" className="space-x-1">
            <FontAwesomeIcon icon={faCalendar} /> <span>Date Denied</span>
          </Text>
        ),
        accessor: 'denied_at',
        Cell: ({ cell: { value } }) => (
          <TableItem center>
            <Text size="sm" className="uppercase">
              {value && value !== 'None' ? formatDate(value) : ''}
            </Text>
          </TableItem>
        ),
      },
      {
        Header: (
          <Text size="sm" className="space-x-1">
            <FontAwesomeIcon icon={faUser} /> <span>Denied by</span>
          </Text>
        ),
        accessor: 'added_by',
        Cell: ({ cell: { value } }) => <Text size="sm">{value}</Text>,
      },
      // {
      //   Header: 'DAA',
      //   accessor: 'daa_document',
      //   Cell: ({cell: {value}}) => (
      //     <TableItem center>
      //       <a href={value}>
      //         <Badge type="subtle" variant="gray">
      //           data_access_agreement.pdf
      //         </Badge>
      //       </a>
      //     </TableItem>
      //   )
      // },
      {
        Header: 'Institution',
        accessor: 'company',
        Cell: ({ cell: { value } }) => (
          <TableItem center>
            <Text size="sm" className="uppercase">
              {value ? formatDate(value) : ''}
            </Text>
          </TableItem>
        ),
      },
      {
        Header: (
          <Text size="sm">
            <FontAwesomeIcon icon={faEnvelope} /> Email
          </Text>
        ),
        accessor: 'email',
        Cell: ({ cell: { value } }) => (
          <a href={`mailto:${value}`}>
            <Text size="sm">{value}</Text>
          </a>
        ),
      },
    ],
    []
  )
  const table = useOMUITable({
    data: tableData,
    columns: tableColumns,
    selectable: true,
    sortable: true,
  })

  const selected = table.instance.selectedFlatRows

  const { open, isOpen, close } = useDisclosure()

  return (
    <section className="space-y-6">
      {/* <div className="flex items-center space-x-2"> */}
      {/*   <Button variant="primary" size="sm" disabled={!selected.length} onClick={open}> */}
      {/*     <Text size="xs" bold> */}
      {/*       Accept ({selected.length}) Users */}
      {/*     </Text> */}
      {/*   </Button> */}
      {/*   <Button */}
      {/*     type="button" */}
      {/*     variant="ghost" */}
      {/*     size="xs" */}
      {/*     disabled={!selected.length} */}
      {/*     onClick={() => table.instance.toggleAllRowsSelected(false)}> */}
      {/*     <Text size="sm" bold className="text-gray-600"> */}
      {/*       Cancel */}
      {/*     </Text> */}
      {/*   </Button> */}
      {/* </div> */}
      {table.Component}
      {/* TODO: support pagination */}
      <Text as="p" size="sm">
        {tableData.length} / {tableData.length} results
      </Text>
      {/* <DeleteModal show={isOpen} onClose={close} /> */}
    </section>
  )
}

function ProcessUser({ id }) {
  const update = useApplicantUsers().update(id).mutate
  return (
    <AcceptDeny
      onAccept={() => update({ status: 'accepted' })}
      onDeny={() => update({ status: 'rejected' })}
    />
  )
}

function PendingUsersTable({ users }) {
  const tableData = useMemo(
    () =>
      users?.map((user) => ({
        ...user,
        summary: {
          name: user.name,
          id: user.id,
          role: user.role,
        },
      })) ?? [],
    [users]
  )

  const tableColumns = useMemo(
    () => [
      {
        Header: 'Name',
        accessor: (d) => d.summary,
        id: 'tab_name',
        Cell: ({ cell: { value } }) => <NameAndBadge {...value} />,
      },
      {
        Header: (
          <Text size="sm">
            <FontAwesomeIcon icon={faCalendar} /> Request Date
          </Text>
        ),
        accessor: 'created_at',
        Cell: ({ cell: { value } }) => (
          <TableItem center>
            <Text size="sm" className="uppercase">
              {value && value !== 'None' ? formatDate(value) : ''}
            </Text>
          </TableItem>
        ),
      },
      // {
      //   Header: 'DAA',
      //   accessor: 'daa_document',
      //   Cell: ({cell: {value}}) => (
      //     <TableItem center>
      //       <a href={value}>
      //         <Badge type="subtle" variant="gray" truncate>
      //           data_access_agreement.pdf
      //         </Badge>
      //       </a>
      //     </TableItem>
      //   )
      // },
      {
        Header: 'Institution',
        accessor: 'company',
        Cell: ({ cell: { value } }) => (
          <TableItem center>
            <Text size="sm" className="uppercase">
              {value ? formatDate(value) : ''}
            </Text>
          </TableItem>
        ),
      },
      {
        Header: (
          <Text size="sm">
            <FontAwesomeIcon icon={faEnvelope} /> Email
          </Text>
        ),
        accessor: 'email',
        Cell: ({ cell: { value } }) => (
          <a href={`mailto:${value}`}>
            <Text size="sm">{value}</Text>
          </a>
        ),
      },
      {
        Header: 'Action',
        accessor: 'id',
        Cell: ({ cell: { value } }) => <ProcessUser id={value} />,
      },
    ],
    []
  )
  const table = useOMUITable({
    data: tableData,
    columns: tableColumns,
    selectable: true,
    sortable: true,
  })

  const selected = table.instance.selectedFlatRows

  const { open, isOpen, close } = useDisclosure()

  return (
    <section className="space-y-6">
      {/* <div className="flex items-center space-x-2"> */}
      {/*   <Button variant="primary" size="sm" disabled={!selected.length} onClick={open}> */}
      {/*     <Text size="xs" bold> */}
      {/*       Accept ({selected.length}) Users */}
      {/*     </Text> */}
      {/*   </Button> */}
      {/*   <Button */}
      {/*     variant="outline" */}
      {/*     className="border-error-500 text-error-500 hover:bg-error-500 hover:text-white" */}
      {/*     size="sm" */}
      {/*     disabled={!selected.length} */}
      {/*     onClick={open}> */}
      {/*     <Text size="xs" bold> */}
      {/*       Deny ({selected.length}) Users */}
      {/*     </Text> */}
      {/*   </Button> */}
      {/*   <Button */}
      {/*     type="button" */}
      {/*     variant="ghost" */}
      {/*     size="xs" */}
      {/*     disabled={!selected.length} */}
      {/*     onClick={() => table.instance.toggleAllRowsSelected(false)}> */}
      {/*     <Text size="sm" bold className="text-gray-600"> */}
      {/*       Cancel */}
      {/*     </Text> */}
      {/*   </Button> */}
      {/* </div> */}
      {table.Component}
      {/* TODO: support pagination */}
      <Text as="p" size="sm">
        {tableData.length} / {tableData.length} results
      </Text>
      <DeleteModal show={isOpen} onClose={close} />
    </section>
  )
}
function ActiveUsersTable({ users, setSelectedUser }) {
  const tableData = useMemo(
    () =>
      users?.map((user) => ({
        ...user,
        summary: {
          name: user.name,
          id: user.id,
          role: user.role,
        },
      })) ?? [],
    [users]
  )

  const tableColumns = useMemo(
    () => [
      {
        Header: 'Name',
        accessor: (d) => d.summary,
        id: 'tab_name',
        Cell: ({ cell: { value, row } }) => (
          <NameAndBadge
            onClick={() => setSelectedUser(row.original)}
            {...value}
          />
        ),
      },
      {
        Header: 'ε Budget Remaining',
        accessor: 'budget_spent',
        Cell: ({ cell: { value, row } }) => {
          const isBudgetRunningOut = value >= row.values.budget * 0.9
          return (
            <TableItem center>
              <Badge
                type={isBudgetRunningOut ? 'solid' : 'subtle'}
                variant={isBudgetRunningOut ? 'danger' : 'gray'}
              >
                {value?.toFixed(2)} ε
              </Badge>
            </TableItem>
          )
        },
      },
      {
        Header: 'ε Allocated Budget',
        accessor: 'budget',
        Cell: ({ cell: { value } }) => (
          <TableItem center>
            <Badge type="subtle" variant="gray">
              {value?.toFixed(2)} ε
            </Badge>
          </TableItem>
        ),
      },
      {
        Header: (
          <Text size="sm">
            <FontAwesomeIcon icon={faCalendar} className="mr-1" /> Date Added
          </Text>
        ),
        accessor: 'created_at',
        Cell: ({ cell: { value } }) => (
          <TableItem center>
            <Text size="sm" className="uppercase">
              {value ? formatDate(value) : ''}
            </Text>
          </TableItem>
        ),
      },
      {
        Header: (
          <Text size="sm">
            <FontAwesomeIcon icon={faUser} className="mr-1" /> Added by
          </Text>
        ),
        id: 'tab_added_by',
        accessor: 'added_by',
        Cell: ({ cell: { value } }) => <NameAndBadge {...value} />,
      },
      {
        Header: (
          <Text size="sm">
            <FontAwesomeIcon icon={faEnvelope} /> Email
          </Text>
        ),
        accessor: 'email',
        Cell: ({ cell: { value } }) => (
          <a href={`mailto:${value}`}>
            <Text size="sm">{value}</Text>
          </a>
        ),
      },
    ],
    []
  )
  const table = useOMUITable({
    data: tableData,
    columns: tableColumns,
    selectable: true,
    sortable: true,
  })

  const selected = table.instance.selectedFlatRows

  const { open, isOpen, close } = useDisclosure()

  return (
    <section className="space-y-6">
      {/* <div className="flex items-center space-x-2"> */}
      {/*   <Button variant="primary" className="bg-error-500" size="sm" disabled={!selected.length} onClick={open}> */}
      {/*     <div className="flex items-center space-x-2"> */}
      {/*       <FontAwesomeIcon icon={faTrash} /> */}
      {/*       <Text size="xs" bold> */}
      {/*         Delete ({selected.length}) Users */}
      {/*       </Text> */}
      {/*     </div> */}
      {/*   </Button> */}
      {/*   <Button */}
      {/*     type="button" */}
      {/*     variant="ghost" */}
      {/*     size="xs" */}
      {/*     disabled={!selected.length} */}
      {/*     onClick={() => table.instance.toggleAllRowsSelected(false)}> */}
      {/*     <Text size="xs" bold className="text-gray-600"> */}
      {/*       Cancel */}
      {/*     </Text> */}
      {/*   </Button> */}
      {/* </div> */}
      {table.Component}
      {/* TODO: support pagination */}
      <Text as="p" size="sm">
        {tableData.length} / {tableData.length} results
      </Text>
      <DeleteModal show={isOpen} onClose={close} />
    </section>
  )
}

function Denied() {
  const { data: users } = useApplicantUsers().all()
  const deniedUsers = useMemo(
    () => users?.filter((user) => user.status === 'rejected'),
    [users]
  )

  return (
    <>
      <div className="col-span-11 mt-10">
        <Alert.Info
          alertStyle="topAccent"
          description={usersStrings.denied.alert}
        />
      </div>
      {/* <div className="flex col-span-3 items-center mt-10"> */}
      {/*   <SearchInput /> */}
      {/* </div> */}
      <Divider color="light" className="col-span-full mt-8" />
      <div className="col-span-full mt-4">
        <DeniedUsersTable users={deniedUsers} />
      </div>
    </>
  )
}

function Pending() {
  const { data: users } = useApplicantUsers().all()
  const pendingUsers = useMemo(
    () => users?.filter((user) => user.status === 'pending'),
    [users]
  )

  return (
    <>
      <div className="col-span-11 mt-10">
        <Alert.Info
          alertStyle="topAccent"
          description={usersStrings.pending.alert}
        />
      </div>
      {/* <div className="flex col-span-3 items-center mt-10"> */}
      {/*   <SearchInput /> */}
      {/* </div> */}
      <Divider color="light" className="col-span-full mt-8" />
      <div className="col-span-full mt-4">
        <PendingUsersTable users={pendingUsers ?? []} />
      </div>
    </>
  )
}

const removeNonNumerical = (value) => String(value).replace(/[^0-9\.]/g, '')

function parseEpsilon(valueWithEpsilon: string | number) {
  const onlyNumerical = removeNonNumerical(valueWithEpsilon)
  const epsilon = parseFloat(onlyNumerical)
  if (epsilon === NaN || epsilon < 0) return 0
  return epsilon
}

function CreateUser({ onClose }) {
  const { mutate: create, isLoading } = useUsers().create(
    { onSuccess: onClose },
    { multipart: true }
  )

  const { register, control, handleSubmit } = useForm({
    defaultValues: {
      name: '',
      email: '',
      password: '',
      confirm_password: '',
      role: 4,
      budget: 10.0,
    },
  })

  const onSubmit = (data) => {
    const formData = new FormData()
    formData.append('new_user', JSON.stringify(data))
    formData.append('file', new Blob())
    create(formData)
  }

  return (
    <>
      <div className="space-y-3 col-span-12">
        <FontAwesomeIcon icon={faUserPlus} className="text-3xl" />
        <H4>Create a User</H4>
      </div>
      <div className="col-span-12 mt-4">
        <form onSubmit={handleSubmit(onSubmit)}>
          <Text size="sm">
            PyGrid utilizes users and roles to appropriately permission data at
            a higher level. All users with the permission Can Create Users are
            allowed to create users in the domain. Create a user by filling out
            the fields below.
          </Text>
          <div className="grid grid-cols-2 gap-6 mt-2.5 mb-12">
            <Input
              {...register('name', { required: true })}
              label="Full name"
              required
            />
            <Input
              {...register('email', { required: true })}
              label="Email"
              required
            />
            <Input
              type="password"
              {...register('password', { required: true })}
              label="Password"
              required
            />
            <Input
              type="password"
              {...register('confirm_password', { required: true })}
              label="Confirm password"
              required
            />
            <div className="col-span-full">
              <Controller
                control={control}
                name="role"
                render={({ field }) => (
                  <Select
                    options={[{ label: 'Data Scientist', value: 4 }]}
                    label="Role"
                    required
                    {...field}
                  />
                )}
              />
            </div>
            <div className="col-span-full space-x-8 flex justify-between">
              <div className="flex-shrink-0">
                <Controller
                  render={({ field: { value, ...rest } }) => (
                    <Input
                      label="Set Privacy Budget (PB)"
                      // type="number"
                      optional
                      addonLeft="-"
                      addonLeftProps={{
                        onClick: () =>
                          rest.onChange((parseEpsilon(value) - 0.1).toFixed(2)),
                      }}
                      addonRight="+"
                      addonRightProps={{
                        onClick: () =>
                          rest.onChange((parseEpsilon(value) + 0.1).toFixed(2)),
                      }}
                      containerProps={{ className: 'max-w-42' }}
                      {...rest}
                      onChange={(e) =>
                        rest.onChange(parseEpsilon(e.target.value).toFixed(2))
                      }
                      value={`${Number(value).toFixed(2)} ε`}
                    />
                  )}
                  name="budget"
                  control={control}
                />
              </div>
              <Text as="p" size="sm">
                Allocating Privacy Budget (PB) is an optional setting that
                allows you to maintain a set standard of privacy while
                offloading the work of manually approving every data request for
                a single user. You can think of privacy budget as credits you
                give to a user to perform computations from. These credits of{' '}
                <Text mono className="text-primary-600" size="xs">
                  Epsilon(ɛ)
                </Text>{' '}
                indicate the amount of visibility a user has into any one entity
                of your data. You can learn more about privacy budgets and how
                to allocate them at{' '}
                <a
                  target="noopener noreferrer _blank"
                  className="text-primary-600 hover:text-primary-500"
                >
                  Course.OpenMined.org
                </a>
              </Text>
            </div>
          </div>
          <div className="col-span-full flex justify-between mt-6 mb-5">
            <Button
              type="button"
              variant="outline"
              onClick={onClose}
              disabled={isLoading}
            >
              Cancel
            </Button>
            <Button variant="primary" type="submit" isLoading={isLoading}>
              <Text bold size="sm">
                <FontAwesomeIcon icon={faPlus} className="mr-1" /> Create
              </Text>
            </Button>
          </div>
        </form>
      </div>
    </>
  )
}

export default function Users() {
  const [currentTab, setCurrentTab] = useState(() => 1)
  const tabsList = [
    { id: 1, title: commonStrings.active_users, disabled: false },
    { id: 2, title: commonStrings.pending_users, disabled: false },
    { id: 3, title: commonStrings.denied_users, disabled: false },
  ]

  return (
    <Base>
      <TopContent
        icon={() => (
          <FontAwesomeIcon icon={sections.users.icon} className="text-3xl" />
        )}
        heading={sections.users.heading}
      />
      <div className="col-span-full">
        <Text size="sm">{sections.users.description}</Text>
      </div>
      <div className="col-span-full mt-14">
        <Tabs
          tabsList={tabsList}
          onChange={setCurrentTab}
          align="auto"
          active={currentTab}
        />
      </div>
      {currentTab === 1 && <Active />}
      {currentTab === 2 && <Pending />}
      {currentTab === 3 && <Denied />}
    </Base>
  )
}
