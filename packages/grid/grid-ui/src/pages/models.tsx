import {useMemo, useState} from 'react'
import {Page, SpinnerWithText, SearchBar, MutationError} from '@/components'
import {ModelList} from '@/components/pages/models'
import {useModels} from '@/lib/data'
import type {Model} from '@/types/grid-types'

interface ModelsList {
  models: Model[]
  isLoading: boolean
  isError: boolean
  errorMessage: string
}

function ModelsList({models, isLoading, isError, errorMessage}: ModelsList) {
  if (isLoading) {
    return <SpinnerWithText>Loading the list of models</SpinnerWithText>
  }

  if (isError) {
    return <MutationError isError error="Unable to load models" description={errorMessage} />
  }

  if (models?.length === 0) {
    return <p>No models are registered in this Domain.</p>
  }

  return <ModelList models={models} />
}

export default function Models() {
  const {all} = useModels()
  const {data: models, isLoading, isError, error} = all()
  const [filtered, setFiltered] = useState<Model[]>(null)
  const searchFields = useMemo(() => ['name', 'description', 'tags', 'id'], [])

  return (
    <Page title="Models" description="Available models">
      <section>
        <SearchBar data={models} searchFields={searchFields} setData={setFiltered} />
      </section>
      <section>
        <ModelsList models={filtered ?? models} isLoading={isLoading} isError={isError} errorMessage={error?.message} />
      </section>
    </Page>
  )
}
