import { createContext, useContext, useMemo, useState } from 'react'
import Link from 'next/link'
import { Badge, Button, Divider, H2, H4, H5, Tabs, Tag, Text } from '@/omui'
import { SearchInput, TopContent } from '@/components/lib'
import { Alert } from '@/components/Alert'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faCalendar, faCheck, faLink, faTimes } from '@fortawesome/free-solid-svg-icons'
import cloneDeep from 'lodash.clonedeep'
import { TableItem, useOMUITable } from '@/components/Table'
import Modal from '@/components/Modal'
import { Base } from '@/components/Layouts'
import { AcceptDeny } from '@/components/AcceptDenyButtons'
import { useDisclosure } from 'react-use-disclosure'
import { formatDate } from '@/utils'
import { useDataRequests, useBudgetRequests, useRequests } from '@/lib/data'
import { RequestStatusBadge } from '@/components/RequestStatusBadge'

const RequestsContext = createContext({ data: [], budget: [], highlighted: null, selected: [] })

function Pending() {
  const { data } = useContext(RequestsContext)
  if (!data) return <EmptyDataRequests />
  return <DataRequestsPendingTable />
}

function DataRequestsPendingTable() {
  const { open, isOpen, close } = useDisclosure()
  const [picked, setPicked] = useState(null)
  const { data } = useContext(RequestsContext)
  const tableData = useMemo(() => data?.filter(req => req?.req?.status === 'pending') ?? [], [data])

  function openDetails(id) {
    setPicked(() => cloneDeep(tableData.find(data => data.req.id === id)))
    open()
  }

  function onClose() {
    setPicked(null)
    close()
  }

  const tableColumns = useMemo(
    () => [
      {
        Header: 'ID#',
        accessor: 'req.id',
        Cell: ({ cell: { value } }) => (
          <Badge variant="gray" type="subtle" truncate className="w-24">
            {value}
          </Badge>
        ),
      },
      {
        Header: 'Name',
        accessor: 'user.name',
        Cell: ({ cell: { value } }) => <Text size="sm">{value}</Text>,
      },
      {
        Header: (
          <Text size="sm" className="space-x-1">
            <FontAwesomeIcon icon={faCalendar} /> <span>Request Date</span>
          </Text>
        ),
        accessor: 'req.date',
        Cell: ({ cell: { value } }) => (
          <Text size="sm" uppercase>
            {formatDate(value)}
          </Text>
        ),
      },
      {
        Header: (
          <Text size="sm" className="space-x-1">
            <FontAwesomeIcon icon={faLink} /> <span>Linked Datasets</span>
          </Text>
        ),
        accessor: 'linked_datasets',
        Cell: ({ cell: { value } }) => (
          <TableItem center>
            {value?.map?.(datasetName => (
              <Badge type="subtle" variant="gray">
                {datasetName}
              </Badge>
            ))}
          </TableItem>
        ),
      },
      {
        Header: 'Request Size',
        accessor: 'req.size',
        Cell: ({ cell: { value } }) => (
          <TableItem center>
            <Badge variant="primary" type="subtle">
              {value} ε
            </Badge>
          </TableItem>
        ),
      },
      {
        Header: 'Action',
        accessor: d => d?.req?.id,
        Cell: ({ cell: { value } }) => {
          const update = useRequests().update(value).mutate
          return (
            <div className="flex space-x-5">
              <button onClick={() => openDetails(value)}>
                <Text underline className="text-primary-600 hover:text-primary-500">
                  See Details
                </Text>
              </button>
              <AcceptDeny
                onAccept={() => update({ status: 'accepted' })}
                onDeny={() => update({ status: 'denied' })}
              />
            </div>
          )
        },
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

  return (
    <>
      {/* <div className="col-span-3 mt-10"> */}
      {/*   <SearchInput /> */}
      {/* </div> */}
      {/* <div className="col-span-full mt-8"> */}
      {/*   <Divider color="light" /> */}
      {/* </div> */}
      <section className="col-span-full space-y-6 mt-4">
        {/* <div className="flex items-center space-x-2"> */}
        {/*   <Button variant="primary" size="sm" disabled={!selected.length} onClick={open}> */}
        {/*     <Text size="xs" bold> */}
        {/*       Accept ({selected.length}) Requests */}
        {/*     </Text> */}
        {/*   </Button> */}
        {/*   <Button */}
        {/*     variant="outline" */}
        {/*     size="sm" */}
        {/*     disabled={!selected.length} */}
        {/*     onClick={open} */}
        {/*     className="border-error-500 text-error-500"> */}
        {/*     <Text size="xs" bold> */}
        {/*       Reject ({selected.length}) Requests */}
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
        {picked && <RequestModal onClose={onClose} show={isOpen} data={picked} />}
      </section>
    </>
  )
}

function DataRequestsHistoryTable() {
  const { open, isOpen, close } = useDisclosure()
  const [picked, setPicked] = useState(null)
  const { data } = useContext(RequestsContext)
  const tableData = useMemo(() => data?.filter(req => req?.req?.status !== 'pending') ?? [], [data])

  function openDetails(id) {
    setPicked(() => cloneDeep(tableData.find(data => data.req.id === id)))
    open()
  }

  const tableColumns = useMemo(
    () => [
      {
        Header: 'ID#',
        accessor: 'req.id',
        Cell: ({ cell: { value } }) => (
          <Badge variant="gray" type="subtle" truncate className="w-24">
            {value}
          </Badge>
        ),
      },
      {
        Header: 'Name',
        accessor: 'user.name',
        Cell: ({ cell: { value } }) => <Text size="sm">{value}</Text>,
      },
      {
        Header: 'Status',
        accessor: 'req.status',
        Cell: ({ cell: { value } }) => <RequestStatusBadge status={value} />,
      },
      {
        Header: (
          <Text size="sm" className="space-x-1">
            <FontAwesomeIcon icon={faCalendar} /> <span>Updated on</span>
          </Text>
        ),
        accessor: 'req.updated_on',
        Cell: ({ cell: { value } }) => (
          <Text size="sm" uppercase>
            {formatDate(value)}
          </Text>
        ),
      },
      {
        Header: (
          <Text size="sm" className="space-x-1">
            <FontAwesomeIcon icon={faCalendar} /> <span>Updated by</span>
          </Text>
        ),
        accessor: 'req.updated_by',
        Cell: ({ cell: { value } }) => (
          <Text size="sm" uppercase>
            {value}
          </Text>
        ),
      },
      {
        Header: 'Request Size',
        accessor: 'req.size',
        Cell: ({ cell: { value, row } }) => (
          <TableItem center>
            <Badge
              variant={row.original.status === 'accepted' ? 'success' : 'danger'}
              type="subtle"
            >
              {value} ε
            </Badge>
          </TableItem>
        ),
      },
      {
        Header: 'Action',
        accessor: d => d?.req?.id,
        Cell: ({ cell: { value } }) => (
          <div onClick={() => openDetails(value)}>
            <Text underline className="text-primary-600 hover:text-primary-500 cursor-pointer">
              See Details
            </Text>
          </div>
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

  return (
    <>
      <div className="col-span-3 mt-10">
        <SearchInput />
      </div>
      <div className="col-span-full mt-8">
        <Divider color="light" className="col-span-full" />
      </div>
      <section className="col-span-full space-y-6 mt-6">
        {table.Component}
        {/* TODO: support pagination */}
        <Text as="p" size="sm">
          {tableData.length} / {tableData.length} results
        </Text>
        {picked && <RequestModal show={isOpen} onClose={close} data={picked} />}
      </section>
    </>
  )
}

function History() {
  const { data } = useContext(RequestsContext)
  if (data?.length === 0) <EmptyDataRequests />
  return <DataRequestsHistoryTable />
}

function RequestModal({ show, onClose, data }) {
  const update = useRequests().update(data?.req?.id).mutate
  const userInformation = [
    {
      text: 'Role',
      value: (
        <Badge variant="primary" type="subtle">
          {data?.user?.role}
        </Badge>
      ),
    },
    {
      text: 'Privacy Budget',
      value: (
        <span className="inline-flex space-x-2">
          <Badge type="solid" variant="danger">
            {Number(data?.user?.budget_spent).toFixed(2)}
          </Badge>
          <Text size="sm">used of</Text>
          <Badge type="subtle" variant="gray">
            {Number(data?.user?.current_budget).toFixed(2)}
          </Badge>
        </span>
      ),
    },
    {
      text: 'Email',
      value: (
        <a href={`mailto:${data?.user?.email}`}>
          <Text size="sm" underline>
            {data?.user?.email}
          </Text>
        </a>
      ),
    },
    { text: 'Company/Institution', value: data?.user?.institution },
    { text: 'Website/Profile', value: data?.user?.website },
  ]
  const requestDetails = [
    { text: 'Request Date', value: data?.req?.date },
    {
      text: 'Tags',
      value: (
        <>
          {data?.req?.tags?.map(tag => (
            <Tag size="sm" tagType="round" variant="primary" className="mr-2">
              {tag}
            </Tag>
          ))}
        </>
      ),
    },
    { text: 'Resource or Result ID', value: data?.req?.result_id },
    // {text: 'Actions', property: 'actions'},
    { text: '# of Values', value: data?.req?.size },
  ]

  return (
    <Modal show={show} onClose={onClose} withExpand>
      <div className="col-span-full grid grid-cols-12">
        <div className="col-span-10 col-start-2 mt-6">
          {/* id */}
          <div className="flex space-x-2 items-center">
            <Text bold size="sm" className="flex-shrink-0">
              Request ID:
            </Text>
            <Badge type="subtle" variant="gray" truncate>
              {data?.req?.id}
            </Badge>
          </div>
          {/* info */}
          <div className="flex justify-between mt-3">
            <H2>{data?.user?.name}</H2>
            <div className="flex space-x-3 flex-shrink-0 items-center">
              <RequestStatusBadge status={data?.req?.status} />
              {data?.req?.status === 'pending' && (
                <>
                  <div className="rounded-full w-7 h-7 bg-gray-200 justify-center items-center flex text-white hover:bg-primary-500 cursor-pointer">
                    <FontAwesomeIcon
                      icon={faCheck}
                      onClick={() => update({ status: 'accepted' })}
                    />
                  </div>
                  <div className="rounded-full w-7 h-7 bg-gray-200 justify-center items-center flex text-white hover:bg-primary-500 cursor-pointer">
                    <FontAwesomeIcon icon={faTimes} onClick={() => update({ status: 'denied' })} />
                  </div>
                </>
              )}
            </div>
          </div>
          {/* cards */}
          <div className="flex space-x-4 w-full mt-4">
            {/* pb-card */}
            <div
              className="w-full bg-gray-50 border border-gray-100 pl-6 pr-8 pt-2 pb-4 space-y-4"
              style={{
                background:
                  'linear-gradient(90deg, rgba(255, 255, 255, 0.8) 0%, rgba(255, 255, 255, 0.5) 100%), #F1F0F4',
              }}
            >
              <div className="flex items-center w-full justify-between">
                <div className="pr-6 border-r border-gray-200">
                  <div className="flex items-center space-x-2">
                    <Text as="p" size="3xl">
                      {data?.req?.size}
                    </Text>
                    <Text size="lg">ɛ</Text>
                  </div>
                  <Text as="p">
                    request size
                    {/* request size <Tooltip position="top">HELT OK</Tooltip> */}
                  </Text>
                </div>
                <div className="pl-6">
                  <Text size="3xl" as="p">
                    {data?.req?.subjects}
                  </Text>
                  <div className="flex items-start space-x-2">
                    <Text as="p">data subjects</Text> {/* <div> */}
                    {/* <Tooltip position="top">NOT OK</Tooltip> */}
                    {/* </div> */}
                  </div>
                </div>
              </div>
              {/* info-card */}
              <Divider color="light" />
              <div>
                <Text bold size="sm">
                  Linked Datasets
                  {/* Linked Datasets <Tooltip position="top">OK</Tooltip> */}
                </Text>
                <div className="flex flex-wrap w-full">
                  {data?.req?.datasets?.map(datasetName => (
                    <div key={datasetName} className="mr-2">
                      <Badge type="subtle" variant="gray" truncate>
                        {datasetName}
                      </Badge>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            {/* request details card */}
            <div className="w-full border border-gray-100 p-6 space-y-4">
              {userInformation.map(info => {
                return (
                  <div key={info.text} className="w-full truncate">
                    <Text size="sm" bold>
                      {info.text}:
                    </Text>{' '}
                    <Text size="sm">{info.value}</Text>
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
          </div>
          <Divider color="light" className="mt-6" />
          <div className="mt-6 space-y-3">
            <H5>Request Details</H5>
            <div className="border border-gray-100 p-6 pb-8 relative">
              <div className="flex flex-col space-y-4">
                <div className="flex items-center">
                  <Text bold>Request ID: </Text>{' '}
                  <Badge variant="gray" type="subtle" truncate>
                    {data?.req?.id}
                  </Badge>
                </div>
                {requestDetails.map(reqDetail => (
                  <div key={reqDetail.text}>
                    <Text bold>{reqDetail.text}:</Text> <Text>{reqDetail.value}</Text>
                  </div>
                ))}
                <Divider color="light" />
                <div className="bg-gray-50 p-3">
                  <Text as="p" bold size="sm">
                    Reason:
                  </Text>
                  <Text as="p" size="sm">
                    {data?.req?.reason}
                  </Text>
                </div>
                {/* <Divider color="light" /> */}
                {/* <div> */}
                {/*   <Button className="w-auto"> */}
                {/*     <Text size="sm" bold> */}
                {/*       <FontAwesomeIcon icon={faDownload} /> Preview Result */}
                {/*     </Text> */}
                {/*   </Button> */}
                {/* </div> */}
                {/* <Text size="sm" as="p"> */}
                {/*   By{' '} */}
                {/*   <Text mono className="text-primary-600" size="sm"> */}
                {/*     Previewing Results */}
                {/*   </Text>{' '} */}
                {/*   you are downloading the results this Data Scientist is requesting. Currently results are in */}
                {/*   [name_here] format. For help viewing the downloaded results you can go here for further instructions. */}
                {/* </Text> */}
              </div>
            </div>
          </div>
        </div>
      </div>
    </Modal>
  )
}

function EmptyDataRequests() {
  return (
    <div className="space-y-2 w-full text-center col-span-8 col-start-3 mt-20">
      <H4>Congratulations</H4>
      <Text className="text-gray-400">You’ve cleared all data requests in your queue!</Text>
    </div>
  )
}

export default function DataRequests() {
  const { data: dataReq } = useDataRequests().all()
  const { data: budgetReq } = useBudgetRequests().all()
  const [currentTab, setCurrentTab] = useState(() => 1)
  const tabsList = [
    { id: 1, title: 'Pending' },
    { id: 2, title: 'History' },
  ]

  return (
    <Base>
      <RequestsContext.Provider
        value={{ data: dataReq, budget: budgetReq, highlighted: null, selected: [] }}
      >
        <TopContent heading="Data Requests" />
        <div className="col-span-10">
          <Alert.Info
            alertStyle="topAccent"
            description="Data requests are one-time requests made from Data Scientists on your node to download the results of their computations. Unlike setting privacy budgets data requests must be manually triaged and do not count as ongoing credits. They are individual allowances based off of specific computations on specified data objects."
          />
        </div>
        <div className="col-span-full mt-10">
          <Tabs tabsList={tabsList} onChange={setCurrentTab} align="auto" active={currentTab} />
        </div>
        {currentTab === 1 && <Pending />}
        {currentTab === 2 && <History />}
      </RequestsContext.Provider>
    </Base>
  )
}
