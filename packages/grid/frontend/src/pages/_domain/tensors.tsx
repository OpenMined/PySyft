import {useMemo, useState} from 'react'
import {Page, SearchBar, SpinnerWithText, MutationError} from '@/components'
import {TensorList} from '@/components/pages/tensors'
import {useTensors} from '@/lib/data'
import type {Tensor} from '@/types/grid-types'

interface TensorsList {
  tensors: Tensor[]
  isLoading: boolean
  isError: boolean
  errorMessage: string
}

function TensorsList({tensors, isLoading, isError, errorMessage}: TensorsList) {
  if (isLoading) {
    return <SpinnerWithText>Loading the list of available tensors</SpinnerWithText>
  }

  if (isError) {
    return <MutationError isError error="Unable to load tensors" description={errorMessage} />
  }

  if (tensors.length === 0) {
    return <p>There are no tensors available in this Domain.</p>
  }

  return <TensorList tensors={tensors} />
}

export default function Tensors() {
  const {all} = useTensors()
  const {data: tensors, isLoading, isError, error} = all()
  const [filtered, setFiltered] = useState<Tensor[]>(null)
  const searchFields = useMemo(() => ['description', 'tags', 'id'], [])

  return (
    <Page title="Tensors" description="List of all available tensors">
      <section>
        <SearchBar data={tensors} searchFields={searchFields} setData={setFiltered} />
      </section>
      <section>
        <TensorsList
          tensors={filtered ?? tensors}
          isLoading={isLoading}
          isError={isError}
          errorMessage={error?.message}
        />
      </section>
    </Page>
  )
}
