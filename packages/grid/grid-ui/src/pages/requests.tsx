import {useMemo, useState} from 'react'
import {Page, SearchBar, SpinnerWithText, MutationError} from '@/components'
import {RequestList} from '@/components/pages/requests'
import {useRequests} from '@/lib/data'
import type {Request} from '@/types/grid-types'

interface RequestsList {
  requests: Request[]
  isLoading: boolean
  isError: boolean
  errorMessage: string
}

function RequestsList({requests, isLoading, isError, errorMessage}: RequestsList) {
  if (isLoading) {
    return <SpinnerWithText>Loading the list of permission requests</SpinnerWithText>
  }

  if (isError) {
    return <MutationError isError error="Unable to load pending permission requests" description={errorMessage} />
  }

  if (requests?.length === 0) {
    return <p>There are no pending permissions requests in this Domain.</p>
  }

  return <RequestList requests={requests} />
}

export default function Requests() {
  const {all} = useRequests()
  const {data: requests, isLoading, isError, error} = all()
  const [filtered, setFiltered] = useState<Request[]>(null)
  const searchFields = useMemo(
    () => ['userName', 'userId', 'objectId', 'objectType', 'status', 'reason', 'tags', 'id'],
    []
  )

  return (
    <Page title="Requests" description="Manage incoming data requests from Data Scientists">
      <section>
        <SearchBar data={requests} searchFields={searchFields} setData={setFiltered} />
      </section>
      <section>
        <RequestsList
          requests={filtered ?? requests}
          isLoading={isLoading}
          isError={isError}
          errorMessage={error?.message}
        />
      </section>
    </Page>
  )
}
