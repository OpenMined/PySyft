import {createContext, useContext, useMemo, useState} from 'react'
import Link from 'next/link'
import {Badge, Button, Divider, H2, H4, ListInnerContainer, Select, Tabs, Text} from '@/omui'
import {Dot, SearchInput, TopContent} from '@/components/lib'
import {Alert} from '@/components/Alert'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {
  faAngleDoubleRight,
  faCalendar,
  faCaretUp,
  faExpandAlt,
  faTimes,
  faUser
} from '@fortawesome/free-solid-svg-icons'
import dayjs from 'dayjs'
import {Accordion} from '@/components/Accordion'
import Modal from '@/components/Modal'
import {Base} from '@/components/Layouts'
import {AcceptDeny} from '@/components/AcceptDenyButtons'
import {formatDate} from '@/utils'
import {useDisclosure} from 'react-use-disclosure'
import {TableItem, useOMUITable} from '@/components/Table'

const RequestsContext = createContext({requests: [], highlighted: null, selected: []})

function PendingUpgrade() {
  const {requests} = useContext(RequestsContext)
  if (requests?.length === 0) <EmptyUpgradeRequests />
  return <RequestsAccordion />
}

function RequestsAccordion() {
  const req = {
    user: {
      name: 'Jane Doe',
      email: 'jane.doe@abc.com',
      role: 'Data Scientist',
      budget: 10,
      used_budget: 10,
      company: 'Oxford University',
      website: 'www.university.edu/reseracher'
    },
    request: {
      id: '12931e4cfdasdf9213nesdf9012#asdASD1',
      current: 10.0,
      requested: 22.0,
      updated_by: {
        name: 'Kyoko Eng',
        role: 'Owner'
      },
      status: 'accepted',
      created_on: dayjs('2021-07-15 08:03:00').format('YYYY-MMM-DD HH:MM'),
      updated_on: dayjs('2021-07-17 09:15:00').format('YYYY-MMM-DD HH:MM'),
      reason:
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
      reviewer_comment: 'Interesting comment'
    }
  }

  const userInformation = [
    {
      text: 'Role',
      value: (
        <Badge variant="primary" type="subtle">
          {req.user.role}
        </Badge>
      )
    },
    {
      text: 'Email',
      value: (
        <a href={`mailto:${req.user.email}`}>
          <Text size="sm" underline>
            {req.user.email}
          </Text>
        </a>
      )
    },
    {text: 'Company/Institution', value: req.user.company},
    {text: 'Website/Profile', value: req.user.website}
  ]

  return (
    <>
      <div className="col-span-3 mt-10">
        <SearchInput />
      </div>
      <div className="col-span-full mt-6">
        <Accordion>
          <Accordion.Item>
            <div className="flex justify-between w-full items-center">
              <div className="items-center flex space-x-2">
                <Text>{req.user.name}</Text>
                <Badge variant="gray" type="subtle">
                  10 ε
                </Badge>
                <Text>+</Text>
                <Badge variant="primary" type="subtle">
                  10 ε
                </Badge>
              </div>
              <div className="flex space-x-2 items-center">
                <Text size="sm">{dayjs(req.request.created_on).fromNow()}</Text>
                <ListInnerContainer>
                  <Dot />
                </ListInnerContainer>
                <AcceptDeny onAccept={() => {}} onDeny={() => {}} />{' '}
              </div>
            </div>
            <div className="flex space-x-6 w-full">
              {/* upgrade-card */}
              <div
                className="w-1/2 flex-shrink-0 bg-gray-50 border border-gray-100 px-6 py-4 items-center"
                style={{
                  background:
                    'linear-gradient(90deg, rgba(255, 255, 255, 0.8) 0%, rgba(255, 255, 255, 0.5) 100%), #F1F0F4'
                }}>
                <div className="flex space-x-6 items-start">
                  <div className="flex-shrink-0">
                    <div className="flex items-center space-x-2 text-error-500">
                      <Text size="xl">{req.request.current.toFixed(2)}</Text>
                      <Text size="lg">ε</Text>
                    </div>
                    <Text size="sm">Current Balance</Text>
                  </div>
                  <div className="self-stretch">
                    <Divider orientation="vertical" color="light" />
                  </div>
                  <div className="flex-shrink-0">
                    <div className="flex items-center space-x-2">
                      <Text size="xl">{req.request.current.toFixed(2)}</Text>
                      <Text size="lg">ε</Text>
                    </div>
                    <Text size="sm">Current Budget</Text>
                  </div>
                </div>
                <Divider color="light" className="mt-8" />
                <div className="flex space-x-4 items-start mt-4">
                  <div className="flex-shrink-0">
                    <div className="flex space-x-2">
                      <div className="flex items-center">
                        <Text size="xl">{req.request.current.toFixed(2)}</Text>
                        <Text size="lg">ε</Text>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Text className="text-primary-600" bold>
                          +
                        </Text>
                        <Badge variant="primary" type="subtle">
                          {req.request.requested - req.request.current} ε
                        </Badge>
                      </div>
                    </div>
                    <Text size="sm">Current Budget</Text>
                  </div>
                  <div className="text-gray-400 w-10 h-10">
                    <FontAwesomeIcon icon={faAngleDoubleRight} className="flex-shrink-0" />
                  </div>
                  <div className="flex-shrink-0">
                    <div className="flex space-x-2">
                      <div flex="flex items-center justify-between">
                        <Text size="xl">{req.request.requested.toFixed(2)}</Text>
                        <Text size="lg">ε</Text>
                      </div>
                      <div flex="flex items-center justify-between">
                        <Badge variant="primary" type="subtle">
                          {req.request.requested - req.request.current}
                          <FontAwesomeIcon icon={faCaretUp} className="pl-1" />
                        </Badge>
                      </div>
                    </div>
                    <Text size="sm">Requested Budget</Text>
                  </div>
                </div>
              </div>
              <div className="flex justify-between space-x-2 truncate">
                {/* request details card */}
                <div className="w-full border border-gray-100 p-6 space-y-4">
                  {userInformation.map(info => {
                    return (
                      <div key={info.property} className="space-x-2 truncate">
                        <Text bold size="sm">
                          {info.text}:
                        </Text>
                        {typeof info.value === 'string' && (
                          <Text size="sm" className="truncate">
                            {info.value}
                          </Text>
                        )}
                        {typeof info.value === 'object' && info.value}
                      </div>
                    )
                  })}
                  <div className="flex flex-col flex-wrap w-full whitespace-normal">
                    <Text bold size="sm">
                      Reason:
                    </Text>
                    <Text size="sm">{req.request.reason}</Text>
                  </div>
                  <div>
                    <Link href="/">
                      <a>
                        <Text as="p" underline size="xs" className="text-primary-600">
                          View User Profile
                        </Text>
                      </a>
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </Accordion.Item>
        </Accordion>
      </div>
    </>
  )
}

function HistoryUpgrade() {
  const {requests} = useContext(RequestsContext)
  return (
    <>
      <div className="col-span-full mt-8">
        {requests?.length === 0 && <EmptyUpgradeRequests />}
        {requests?.length > 0 && <UpgradeRequestsHistoryTable />}
      </div>
    </>
  )
}

function UpgradeRequestsHistoryTable() {
  const {requests} = useContext(RequestsContext)
  const tableData = useMemo(() => requests ?? [], [requests])

  const tableColumns = useMemo(
    () => [
      {
        Header: 'ID#',
        accessor: 'id',
        Cell: ({cell: {value}}) => <Text size="sm">{value}</Text>
      },
      {
        Header: 'Name',
        accessor: 'name',
        Cell: ({cell: {value}}) => <Text size="sm">{value}</Text>
      },
      {
        Header: 'Status',
        accessor: 'status',
        Cell: ({cell: {value}}) => <Text size="sm">{value}</Text>
      },
      {
        Header: (
          <Text size="sm" className="space-x-1">
            <FontAwesomeIcon icon={faCalendar} /> <span>Updated on</span>
          </Text>
        ),
        accessor: 'updated_on',
        Cell: ({cell: {value}}) => (
          <Text size="sm" uppercase>
            {formatDate(value)}
          </Text>
        )
      },
      {
        Header: (
          <Text size="sm" className="space-x-1">
            <FontAwesomeIcon icon={faUser} /> <span>Updated by</span>
          </Text>
        ),
        accessor: 'updated_by',
        Cell: ({cell: {value}}) => (
          <Text size="sm" uppercase>
            {value}
          </Text>
        )
      },
      {
        Header: 'Requested',
        accessor: 'requested_pb',
        Cell: ({cell: {value, row}}) => (
          <TableItem center>
            <Badge variant={row.original.status === 'accepted' ? 'success' : 'danger'} type="subtle">
              {value} ε
            </Badge>
          </TableItem>
        )
      },
      {
        Header: 'New budget',
        accessor: 'new_pb',
        Cell: ({cell: {value, row}}) => (
          <TableItem className="h-full flex">
            <div className="flex items-center border-r pr-3">
              <Badge variant="gray" type="subtle">
                {value} ε
              </Badge>
            </div>
            <div className="flex items-center ml-3">
              <Text size="sm" className="text-success-600">
                {value - row.original.requested_pb}
              </Text>
            </div>
          </TableItem>
        )
      }
    ],
    []
  )
  const table = useOMUITable({
    data: tableData,
    columns: tableColumns,
    selectable: true,
    sortable: true
  })

  const selected = table.instance.selectedFlatRows

  const {open, isOpen, close} = useDisclosure()

  return (
    <>
      <div className="flex col-span-full mt-10">
        <SearchInput />
        <Dot />
        <Select placeholder="Filter by Status" />
      </div>
      <section className="col-span-full space-y-6">
        {table.Component}
        {/* TODO: support pagination */}
        <Text as="p" size="sm">
          {tableData.length} / {tableData.length} results
        </Text>
      </section>
    </>
  )
}

function RequestUpgradeModal() {
  const req = {
    user: {
      name: 'Jane Doe',
      email: 'jane.doe@abc.com',
      role: 'Data Scientist',
      budget: 10,
      used_budget: 10,
      company: 'Oxford University',
      website: 'www.university.edu/reseracher'
    },
    request: {
      id: '12931e4cfdasdf9213nesdf9012#asdASD1',
      current: 10.0,
      requested: 22.0,
      updated_by: {
        name: 'Kyoko Eng',
        role: 'Owner'
      },
      status: 'accepted',
      created_on: dayjs('2021-07-15 08:03:00').format('YYYY-MMM-DD HH:MM'),
      updated_on: dayjs('2021-07-17 09:15:00').format('YYYY-MMM-DD HH:MM'),
      reason:
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
      reviewer_comment: 'Interesting comment'
    }
  }

  const userInformation = [
    {text: 'Role', property: 'role'},
    {text: 'Privacy Budget', property: 'used_budget'},
    {text: 'Email', property: 'email'},
    {text: 'Company/Institution', property: 'company'},
    {text: 'Website/Profile', property: 'website'}
  ]

  const systemInformation = [
    {
      text: 'Request ID',
      value: (
        <Badge variant="gray" type="subtle" truncate>
          {req.request.id}
        </Badge>
      )
    },
    {
      text: 'Status',
      value: (
        <Badge variant="success" type="solid" truncate className="capitalize">
          {req.request.status}
        </Badge>
      )
    },
    {
      text: 'Updated by',
      value: (
        <div className="space-x-2 flex w-full">
          <Text>{req.request.updated_by.name}</Text>
          <Badge variant="primary" type="subtle" truncate>
            {req.request.updated_by.role}
          </Badge>
        </div>
      )
    },
    {text: 'Updated on', value: req.request.updated_on},
    {text: 'Request date', value: req.request.created_on},
    {text: 'Reviewer comment', value: req.request.reviewer_comment}
  ]

  return (
    <Modal show withExpand>
      <div className="col-span-full grid grid-cols-12">
        <div className="col-span-10 col-start-2 mt-6">
          {/* id */}
          <div className="flex space-x-2 items-center">
            <Text bold size="sm" className="flex-shrink-0">
              Request ID:
            </Text>
            <Badge type="subtle" variant="gray" truncate>
              {req.request.id}
            </Badge>
          </div>
          {/* info */}
          <div className="flex justify-between items-center mt-3">
            <H2>{req.user.name}</H2>
            <Badge variant="success" type="solid" className="capitalize">
              {req.request.status}
            </Badge>
          </div>
          {/* cards */}
          <div className="flex space-x-6 w-full mt-4">
            {/* upgrade-card */}
            <div
              className="w-1/2 bg-gray-50 border border-gray-100 px-6 py-4 items-center"
              style={{
                background:
                  'linear-gradient(90deg, rgba(255, 255, 255, 0.8) 0%, rgba(255, 255, 255, 0.5) 100%), #F1F0F4'
              }}>
              <div className="flex space-x-4 items-start">
                <div className="flex-shrink-0">
                  <div className="flex space-x-2">
                    <div className="flex items-center">
                      <Text size="xl">{req.request.current.toFixed(2)}</Text>
                      <Text size="lg">ε</Text>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Text className="text-success-600" bold>
                        +
                      </Text>
                      <Badge variant="success" type="solid">
                        {req.request.requested - req.request.current} ε
                      </Badge>
                    </div>
                  </div>
                  <Text size="sm">Current Budget</Text>
                </div>
                <div className="text-gray-400 w-10 h-10">
                  <FontAwesomeIcon icon={faAngleDoubleRight} className="flex-shrink-0" />
                </div>
                <div className="flex-shrink-0">
                  <div className="flex space-x-2">
                    <div flex="flex items-center justify-between">
                      <Text size="xl">{req.request.requested.toFixed(2)}</Text>
                      <Text size="lg">ε</Text>
                    </div>
                    <div flex="flex items-center justify-between">
                      <Badge variant="success" type="solid">
                        {req.request.requested - req.request.current}
                        <FontAwesomeIcon icon={faCaretUp} className="pl-1" />
                      </Badge>
                    </div>
                  </div>
                  <Text size="sm">Requested Budget</Text>
                </div>
              </div>
            </div>
            {/* Reason */}
            <div className="w-1/2 flex flex-col justify-center">
              <Text bold size="sm" as="p">
                Reason:
              </Text>
              <Text size="sm" as="p">
                {req.request.reason}
              </Text>
            </div>
          </div>
          <Divider color="light" className="my-10" />
          <div className="flex justify-between space-x-2">
            {/* request details card */}
            <div className="w-full border border-gray-100 p-6 space-y-4">
              {userInformation.map(info => {
                return (
                  <div key={info.property} className="space-x-2">
                    <Text bold size="sm">
                      {info.text}:
                    </Text>
                    <Text size="sm">{req.user[info.property]}</Text>
                  </div>
                )
              })}
              <div>
                <Link href="/">
                  <a>
                    <Text as="p" underline size="xs" className="text-primary-600">
                      View User Profile
                    </Text>
                  </a>
                </Link>
              </div>
            </div>
            {/* system details card */}
            <div className="w-1/2 flex-shrink-0 border border-gray-100 p-6 space-y-4">
              {systemInformation.map(info => {
                return (
                  <div key={info.text} className="space-x-2 flex items-center">
                    <Text bold size="sm" className="flex-shrink-0" as="p">
                      {info.text}:
                    </Text>
                    {typeof info.value === 'string' && (
                      <Text size="sm" className="truncate" as="p">
                        {info.value}
                      </Text>
                    )}
                    {typeof info.value === 'object' && info.value}
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </Modal>
  )
}

function EmptyUpgradeRequests() {
  return (
    <div className="space-y-2 w-full text-center mt-20">
      <H4>Congratulations</H4>
      <Text className="text-gray-400">You’ve cleared all upgrade requests in your queue!</Text>
    </div>
  )
}

export default function UpgradeRequests() {
  const [currentTab, setCurrentTab] = useState(() => 1)
  const tabsList = [
    {id: 1, title: 'Pending'},
    {id: 2, title: 'History'}
  ]
  const requests = []

  return (
    <Base>
      <RequestsContext.Provider value={{requests, highlighted: null, selected: []}}>
        <TopContent heading="Upgrade Requests" />
        <div className="col-span-10">
          <Alert.Info
            alertStyle="topAccent"
            description={
              <Text className="text-gray-800">
                Upgrade requests are requests made by Data Scientists on your node to get a larger amount of privacy
                budget allocated to them. You can think of privacy budget as credits you give to a user to perform
                computations from. These credits of{' '}
                <Text mono className="text-gray-600">
                  Epsilon(ɛ)
                </Text>{' '}
                indicate the amount of visibility a user has into any one entity of your data.
              </Text>
            }
          />
        </div>
        <div className="col-span-full mt-10">
          <Tabs tabsList={tabsList} onChange={setCurrentTab} align="auto" active={currentTab} />
        </div>
        {currentTab === 1 && <PendingUpgrade />}
        {currentTab === 2 && <HistoryUpgrade />}
        <RequestUpgradeModal />
      </RequestsContext.Provider>
    </Base>
  )
}
