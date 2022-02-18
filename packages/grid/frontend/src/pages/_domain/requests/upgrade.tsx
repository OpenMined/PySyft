import {createContext, useContext, useMemo, useState} from 'react'
import {Badge, Divider, H4, ListInnerContainer, Select, Tabs, Text} from '@/omui'
import {Dot, SearchInput, TopContent} from '@/components/lib'
import {Alert} from '@/components/Alert'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {faAngleDoubleRight, faCaretUp} from '@fortawesome/free-solid-svg-icons'
import dayjs from 'dayjs'
import {Accordion} from '@/components/Accordion'
import {Base} from '@/components/Layouts'
import {AcceptDeny} from '@/components/AcceptDenyButtons'
import {TableItem, useOMUITable} from '@/components/Table'
import {useBudgetRequests, useRequests} from '@/lib/data'
import {RequestStatusBadge} from '@/components/RequestStatusBadge'

const RequestsContext = createContext({budgets: [], requests: [], highlighted: null, selected: []})

function PendingUpgrade() {
  const {budgets} = useContext(RequestsContext)
  const pending = budgets?.filter(b => b?.req?.status === 'pending')
  if (pending?.length === 0) return <EmptyUpgradeRequests />
  return <RequestsAccordion budgets={pending} />
}

function RequestsAccordion({budgets}) {
  const buildUserInfo = info => [
    {
      text: 'Role',
      value: (
        <Badge variant="primary" type="subtle">
          {info.user.role}
        </Badge>
      )
    },
    {
      text: 'Email',
      value: (
        <a href={`mailto:${info.user.email}`}>
          <Text size="sm" underline>
            {info.user.email}
          </Text>
        </a>
      )
    },
    {text: 'Company/Institution', value: info.user.company},
    {text: 'Website/Profile', value: info.user.website}
  ]

  return (
    <>
      <div className="col-span-3 mt-10">
        <SearchInput />
      </div>
      <div className="col-span-full mt-6">
        <Accordion>
          {budgets?.map(item => {
            const update = useRequests().update(item.req.id).mutate
            return (
              <Accordion.Item>
                <div className="flex justify-between w-full items-center">
                  <div className="items-center flex space-x-2">
                    <Text>{item.user?.name}</Text>
                    <Text size="xs">
                      (<a href={`mailto:${item.user.email}`}>{item.user.email}</a>)
                    </Text>
                    <Badge size="sm" variant="primary" type="subtle" truncate>
                      {item.user.role}
                    </Badge>
                    <Badge variant="primary" type="subtle">
                      {item.req.requested_budget} ε
                      <FontAwesomeIcon icon={faCaretUp} className="pl-1" />
                    </Badge>
                  </div>
                  <div className="flex space-x-2 items-center">
                    <Text size="sm">{dayjs(item.req.date).fromNow()}</Text>
                    <ListInnerContainer>
                      <Dot />
                    </ListInnerContainer>
                    <AcceptDeny
                      onAccept={() => update({status: 'accepted'})}
                      onDeny={() => update({status: 'denied'})}
                    />{' '}
                  </div>
                </div>
                <div className="flex space-x-6 w-full">
                  {/* upgrade-card */}
                  <div
                    className="flex-shrink-0 bg-gray-50 border border-gray-100 px-4 py-0 items-center"
                    style={{
                      background:
                        'linear-gradient(90deg, rgba(255, 255, 255, 0.8) 0%, rgba(255, 255, 255, 0.5) 100%), #F1F0F4'
                    }}>
                    <div className="flex space-x-4 items-start mt-4">
                      <div className="flex-shrink-0">
                        <div className="flex space-x-2">
                          <div className="flex items-center">
                            <Text size="xl">{item.user.current_budget.toFixed(2)}</Text>
                            <Text size="lg">ε</Text>
                          </div>
                          <div className="flex items-center space-x-1">
                            <Text className="text-primary-600" bold>
                              +
                            </Text>
                            <Badge variant="primary" type="subtle">
                              {item.req.requested_budget} ε
                            </Badge>
                          </div>
                        </div>
                        <Text size="sm">Current Balance</Text>
                      </div>
                      <div className="text-gray-400 w-10 h-10">
                        <FontAwesomeIcon icon={faAngleDoubleRight} className="flex-shrink-0" />
                      </div>
                      <div className="flex-shrink-0">
                        <div className="flex space-x-2">
                          <div className="flex items-center justify-between text-error-500">
                            <Text size="xl">{(item.req.requested_budget + item.user.current_budget).toFixed(2)}</Text>
                            <Text size="lg">ε</Text>
                          </div>
                          <div className="flex items-center justify-between">
                            <Badge variant="primary" type="subtle">
                              {item.req.requested_budget} ε
                              <FontAwesomeIcon icon={faCaretUp} className="pl-1" />
                            </Badge>
                          </div>
                        </div>
                        <Text size="sm">New Budget</Text>
                      </div>
                    </div>
                  </div>
                  <div className="flex space-x-2 truncate w-full">
                    {/* request details card */}
                    <div className="w-full p-6 space-y-4">
                      {() => {
                        const info = buildUserInfo(item)
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
                      }}
                      <div className="flex flex-col flex-wrap w-full whitespace-normal">
                        <Text bold size="sm">
                          Reason:
                        </Text>
                        <Text size="sm">{item.req.reason}</Text>
                      </div>
                      {/* <div> */}
                      {/*   <Link href="/"> */}
                      {/*     <a> */}
                      {/*       <Text as="p" underline size="xs" className="text-primary-600"> */}
                      {/*         View User Profile */}
                      {/*       </Text> */}
                      {/*     </a> */}
                      {/*   </Link> */}
                      {/* </div> */}
                    </div>
                  </div>
                </div>
              </Accordion.Item>
            )
          })}
        </Accordion>
      </div>
    </>
  )
}

function HistoryUpgrade() {
  const {budgets} = useContext(RequestsContext)
  const history = budgets?.filter(req => req.req.status !== 'pending')
  return (
    <div className="col-span-full mt-8">
      {history?.length === 0 && <EmptyUpgradeRequests />}
      {history?.length > 0 && <UpgradeRequestsHistoryTable />}
    </div>
  )
}

function UpgradeRequestsHistoryTable() {
  const {budgets} = useContext(RequestsContext)
  const tableData = useMemo(() => budgets?.filter(req => req.req.status !== 'pending') ?? [], [budgets])
  const tableColumns = useMemo(
    () => [
      {
        Header: 'ID#',
        accessor: 'req.id',
        Cell: ({cell: {value}}) => (
          <Badge size="sm" variant="gray" type="subtle" truncate>
            {value}
          </Badge>
        )
      },
      {
        Header: 'Name',
        accessor: 'user.name',
        Cell: ({cell: {value}}) => <Text size="sm">{value}</Text>
      },
      {
        Header: 'Status',
        accessor: 'req.status',
        Cell: ({cell: {value}}) => <RequestStatusBadge status={value} />
      },
      // {
      //   Header: (
      //     <Text size="sm" className="space-x-1">
      //       <FontAwesomeIcon icon={faCalendar} /> <span>Updated on</span>
      //     </Text>
      //   ),
      //   accessor: 'req.updated_on',
      //   Cell: ({cell: {value}}) => (
      //     <Text size="sm" uppercase>
      //       {formatDate(value)}
      //     </Text>
      //   )
      // },
      // {
      //   Header: (
      //     <Text size="sm" className="space-x-1">
      //       <FontAwesomeIcon icon={faUser} /> <span>Updated by</span>
      //     </Text>
      //   ),
      //   accessor: 'req.updated_by',
      //   Cell: ({cell: {value}}) => (
      //     <Text size="sm" uppercase>
      //       {value}
      //     </Text>
      //   )
      // },
      {
        Header: 'Requested',
        accessor: 'req.requested_budget',
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
        accessor: '',
        Cell: ({cell: {value, row}}) => (
          <TableItem className="h-full flex">
            <div className="flex items-center border-r pr-3">
              <Badge variant="gray" type="subtle">
                {row.original.req.status === 'denied'
                  ? row.original.user.current_budget
                  : row.original.req.requested_budget}{' '}
                ε
              </Badge>
            </div>
            <div className="flex items-center ml-3">
              <Text size="sm" className={row.original.req.status === 'denied' ? 'text-error-600' : 'text-success-600'}>
                {row.original.req.status === 'denied'
                  ? '--'
                  : `+${row.original.req.requested_budget - row.original.user.current_budget}`}
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

  // const selected = table.instance.selectedFlatRows

  return (
    <>
      {/* <div className="flex col-span-full mt-10"> */}
      {/*   <SearchInput /> */}
      {/*   <Dot /> */}
      {/*   <Select placeholder="Filter by Status" /> */}
      {/* </div> */}
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

function EmptyUpgradeRequests() {
  return (
    <div className="col-span-full space-y-2 w-full text-center mt-20">
      <H4>Congratulations</H4>
      <Text className="text-gray-400">You’ve cleared all upgrade requests in your queue!</Text>
    </div>
  )
}

export default function UpgradeRequests() {
  const {data: budgetReq} = useBudgetRequests().all()
  const [currentTab, setCurrentTab] = useState(() => 1)
  const tabsList = [
    {id: 1, title: 'Pending'},
    {id: 2, title: 'History'}
  ]
  const requests = []

  return (
    <Base>
      <RequestsContext.Provider value={{budgets: budgetReq, requests, highlighted: null, selected: []}}>
        <TopContent heading="Upgrade Requests" />
        <div className="col-span-10">
          <Alert.Info
            alertStyle="topAccent"
            description={
              <Text className="text-gray-800">
                Upgrade requests are requests made by Data Scientists on your node to get a larger amount of privacy
                budget allocated to them. You can think of privacy budget as credits you give to a user to perform
                computations with. These credits of{' '}
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
      </RequestsContext.Provider>
    </Base>
  )
}
