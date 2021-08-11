import {useMemo, useState} from 'react'
import {useRouter} from 'next/router'
import {Page, NormalButton, SearchBar, SpinnerWithText, MutationError, Alert} from '@/components'
import {DatasetCreatePanel, DatasetList} from '@/components/pages/datasets'
import {formatFullDate} from '@/utils'
import {useDatasets} from '@/lib/data'
import {useEnhancedCurrentUser} from '@/lib/users/self'
import {ViewDataset} from '@/components/pages/datasets/ViewDataset'
import type {Dataset} from '@/types/grid-types'

function ViewOne({id}: {id: string}) {
  const router = useRouter()
  const {one} = useDatasets()
  const {data: dataset, isError, isLoading} = one(id)

  if (isError) {
    router.push('/datasets')
    return
  }

  if (isLoading) {
    return (
      <Page>
        <SpinnerWithText>Loading dataset {id}</SpinnerWithText>
      </Page>
    )
  }

  if (dataset) {
    return (
      <Page title={dataset.name || 'Unnamed dataset'} description={`Created at ${formatFullDate(dataset.createdAt)}`}>
        <ViewDataset dataset={dataset} />
      </Page>
    )
  }

  return (
    <Page title="Datasets">
      <Alert error="Something went wrong..." description="Please contact support via the navigation bar" />
    </Page>
  )
}

function DatasetsList({datasets, isLoading, isError, error}) {
  if (isLoading) return <SpinnerWithText>Loading all available datasets</SpinnerWithText>
  if (isError) return <MutationError isError error="Unable to load datasets" description={error?.message} />
  if (datasets.length === 0) return <p>There are no datasets in this Domain.</p>
  return <DatasetList datasets={datasets} />
}

function ViewAll() {
  const {all} = useDatasets()
  const {data: datasets, isLoading, isError, error} = all()
  const me = useEnhancedCurrentUser()

  const [openCreatePanel, setOpen] = useState<boolean>(false)
  const [filtered, setFiltered] = useState<Dataset[]>(null)

  const searchFields = useMemo(() => ['name', 'description', 'manifest', 'tags', 'id'], [])

  return (
    <Page title="Datasets" description="Available datasets">
      <section className="flex justify-between space-x-6">
        <SearchBar data={datasets} searchFields={searchFields} setData={setFiltered} />
        {me?.permissions?.canUploadData && (
          <NormalButton
            className={`flex-shrink-0 bg-gray-900 text-gray-50 bg-opacity-80 hover:bg-opacity-100`}
            onClick={() => setOpen(true)}>
            Create dataset
          </NormalButton>
        )}
      </section>
      {openCreatePanel && (
        <section>
          <DatasetCreatePanel onClose={() => setOpen(false)} />
        </section>
      )}
      <section>
        <DatasetsList datasets={filtered ?? datasets} isLoading={isLoading} isError={isError} error={error?.message} />
      </section>
    </Page>
  )
}

export default function Datasets() {
  const router = useRouter()

  if (router?.query?.d) {
    return <ViewOne id={router.query.d as string} />
  }

  return <ViewAll />
}
