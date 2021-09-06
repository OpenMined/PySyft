import {XCircleIcon, CheckCircleIcon} from '@heroicons/react/solid'
import dayjs from 'dayjs'
import {Badge, Accordion, Spinner} from '@/components'
import {dateFromNow, statusColors, formatFullDate} from '@/utils'
import {useRequests} from '@/lib/data'
import type {MouseEventHandler} from 'react'
import type {Request, RequestStatus} from '@/types/grid-types'

interface ListType {
  requests: Request[]
  listType: RequestStatus
}

export function RequestList({requests}: {requests: Request[]}) {
  const types: RequestStatus[] = ['pending', 'accepted', 'denied']
  return (
    <div className="space-y-6">
      {types.map(type => (
        <RequestListType key={`${type}-request-list`} requests={requests} listType={type} />
      ))}
    </div>
  )
}

function RequestListType({requests, listType}: ListType) {
  const orderedRequests = requests.sort((a, b) => dayjs(b.date).valueOf() - dayjs(a.date).valueOf())
  const list = orderedRequests.filter(request => request.status === listType)

  return (
    <div className="space-y-2">
      <h2 className="flex items-center space-x-3">
        <Badge bgColor="bg-white" textColor="text-gray-800" className="py-2 text-sm tracking-tight">
          {listType}
          {list.length >= 0 && (
            <div className="h-full px-2 rounded-md inline-block bg-yellow-400 text-gray-800 font-semibold text-xs py-0.5 ml-2">
              {list.length}
            </div>
          )}
        </Badge>
      </h2>
      <Accordion>
        {list.map(item => (
          <Accordion.Item key={item.id}>
            <div className="flex items-center justify-between w-full">
              <RequestAccordionTitle {...item} />
              <RequestAccordionActions {...item} />
            </div>
            <RequestInfoPanel {...item} />
          </Accordion.Item>
        ))}
      </Accordion>
    </div>
  )
}

function RequestAccordionActions(request: Request) {
  return (
    <div className="flex items-center space-x-4">
      {request.status === 'pending' && <AcceptOrDenyPermissionRequest {...request} />}
      <RequestStatusAndDate {...request} />
    </div>
  )
}

function RequestStatusAndDate({status, date}: Pick<Request, 'status' | 'date'>) {
  return (
    <div className="flex flex-col items-end justify-center flex-shrink-0 space-y-1 lg:flex-row lg:justify-start lg:items-center lg:space-x-2 lg:space-y-0 ">
      <div className="w-20">
        <Badge className="w-auto" bgColor={statusColors.requests[status]}>
          {status}
        </Badge>
      </div>
      <div className="flex-shrink-0 w-24 text-sm text-right text-gray-500">{dateFromNow(date)}</div>
    </div>
  )
}

function RequestAccordionTitle(request: Request) {
  return (
    <div className="text-left">
      <p>
        <span className="font-medium">{request.userName}</span> needs permission
      </p>
      <p className="text-xs text-gray-500">{request.tags?.join(', ')}</p>
    </div>
  )
}

function RequestInfoPanel(request: Request) {
  return (
    <div className="py-6 pl-16 pr-4 space-y-6 text-sm border-t border-gray-100 bg-blueGray-100">
      <PermissionRequestInfo {...request} />
      <PermissionObjectInfo {...request} />
      <PermissionReason {...request} />
      <PermissionTags {...request} />
    </div>
  )
}

function AcceptOrDenyPermissionRequest({id}: Pick<Request, 'id'>) {
  const {update} = useRequests()
  const mutation = update(id)

  const handleAccept: MouseEventHandler<HTMLButtonElement> = e => {
    e.stopPropagation()
    mutation.mutate({status: 'accepted'})
  }

  const handleDeny: MouseEventHandler<HTMLButtonElement> = e => {
    e.stopPropagation()
    mutation.mutate({status: 'denied'})
  }

  if (mutation.isLoading) {
    return <Spinner className="w-6 h-6 text-gray-400" />
  }

  return (
    <div className="flex items-center space-x-2 text-gray-300">
      <button onClick={handleAccept}>
        <CheckCircleIcon className="w-8 h-8 hover:text-green-600" />
      </button>
      <button onClick={handleDeny}>
        <XCircleIcon className="w-8 h-8 hover:text-red-600" />
      </button>
    </div>
  )
}

function PermissionRequestInfo({userName, date, id}: Pick<Request, 'userName' | 'date' | 'id'>) {
  return (
    <div className="space-y-1">
      <h3 className="mb-1 text-base font-medium">Permission request</h3>
      <p>
        Requested by {userName} on {formatFullDate(date)}
      </p>
      <p className="text-sm text-gray-500">Request id #{id}</p>
    </div>
  )
}

function PermissionObjectInfo({objectId, objectType}: Pick<Request, 'objectId' | 'objectType'>) {
  return (
    <div>
      <h3 className="mb-1 text-base font-medium">Object information</h3>
      <div className="space-y-2">
        <p>Object id </p>
        <div className="px-2 py-1 text-sm bg-gray-50">{objectId}</div>
        <p>Object type</p>
        <div className="px-2 py-1 text-sm bg-gray-50">{objectType}</div>
      </div>
    </div>
  )
}

function PermissionReason({reason}: Pick<Request, 'reason'>) {
  return (
    <div className="space-y-2">
      <h3 className="text-base font-medium">Reason</h3>
      <p className="p-2 text-sm border-l-8 rounded-l-sm border-amber-400 bg-amber-100">{reason}</p>
    </div>
  )
}

function PermissionTags({tags}: Pick<Request, 'tags'>) {
  return (
    <div className="space-y-2">
      <h3 className="text-base font-medium">Tags</h3>
      <p>{tags?.join(' ')}</p>
    </div>
  )
}
