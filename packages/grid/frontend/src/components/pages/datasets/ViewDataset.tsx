import { createContext, useContext, useState } from 'react'
import { useRouter } from 'next/router'
import cn from 'classnames'
import { useForm } from 'react-hook-form'
import { XCircleIcon } from '@heroicons/react/outline'
import {
  Input,
  DeleteButton,
  NormalButton,
  Tag,
  Table,
  TableData,
  TableRow,
} from '@/components'
import { useDatasets } from '@/lib/data'
import type { ReactNode, ChangeEventHandler } from 'react'
import type { Dataset } from '@/types/grid-types'
import { useEnhancedCurrentUser } from '@/lib/users/self'

function Title({ children }) {
  return <h2 className="text-xl font-semibold capitalize">{children}</h2>
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-2">
      <Title>{title}</Title>
      {children}
    </section>
  )
}

function AddTag() {
  const { id, tags } = useContext(DatasetContext)
  const [tag, setTag] = useState('')
  const [error, setError] = useState('')
  const { update } = useDatasets()
  const mutation = update(id, { mutationKey: 'dataset-add-tag' })

  const canAddTag = tag && tags.indexOf(`#${tag}`) === -1

  const handleChange: ChangeEventHandler<HTMLInputElement> = (event) => {
    if (error) {
      setError('')
    }

    setTag(event.target.value)
  }

  const handleAdd = () => {
    if (!canAddTag) {
      setError('Tag already exists')
      return
    }

    mutation.mutate(
      { tags: [...tags, `#${tag.replace(' ', '-')}`] },
      { onSuccess: () => setTag('') }
    )
  }

  if (!id) {
    return null
  }

  return (
    <form className="flex space-x-4">
      <Input
        pre="#"
        label="Add a new tag"
        id="add-tag"
        placeholder="Tag name"
        className="rounded-l-none"
        onChange={handleChange}
        name="add"
        value={tag}
        error={error || mutation?.error?.message}
      />
      <div
        className={cn('mt-auto', (error || mutation?.error?.message) && 'mb-5')}
      >
        <NormalButton
          type="button"
          className="w-16"
          isLoading={mutation.isLoading}
          disabled={!tag || !canAddTag}
          onClick={handleAdd}
        >
          Add
        </NormalButton>
      </div>
    </form>
  )
}

function TagList() {
  const { id, tags } = useContext(DatasetContext)
  const { update } = useDatasets()
  const mutation = update(id, { mutationKey: 'dataset-tag-remove' })

  const deleteTag = (tag: string) =>
    mutation.mutate({ tags: tags.filter((t) => t !== tag) })

  if (!id) {
    return null
  }

  if (tags?.length === 1) {
    return (
      <div>
        <Tag className="mb-2 bg-blueGray-50">{tags[0]}</Tag>
      </div>
    )
  }

  return (
    <div className="flex flex-wrap">
      {tags?.map((tag) => (
        <div key={`${id}-${tag}`} className="flex mb-2 mr-2 group">
          <Tag className="rounded-r-none bg-pink-50">{tag}</Tag>
          <button
            className={cn(
              'cursor-pointer px-2 rounded-r-md text-white items-center flex bg-red-400',
              'hover:bg-red-800',
              'disabled:bg-gray-300 disabled:cursor-not-allowed'
            )}
            disabled={mutation.isLoading}
            onClick={() => deleteTag(tag)}
          >
            <XCircleIcon className="w-4 h-4" />
          </button>
        </div>
      ))}
    </div>
  )
}

function ChangeDatasetName() {
  const { register, handleSubmit } = useForm({ mode: 'onTouched' })
  const { id, name: datasetName } = useContext(DatasetContext)
  const { update } = useDatasets()
  const mutation = update(id, { mutationKey: 'dataset-name' })

  const onSubmit = ({ name }) => mutation.mutate({ name })

  return (
    <form className="flex w-full space-x-4" onSubmit={handleSubmit(onSubmit)}>
      <Input
        id="dataset-name"
        name="name"
        label="Dataset name"
        container="w-full"
        placeholder="Insert name"
        defaultValue={datasetName}
        ref={register}
      />
      <NormalButton
        className="mt-auto w-28"
        disabled={mutation.isLoading}
        isLoading={mutation.isLoading}
      >
        Change
      </NormalButton>
    </form>
  )
}

function DeleteDataset() {
  const router = useRouter()
  const { id } = useContext(DatasetContext)
  const { remove } = useDatasets()
  const mutation = remove(id)
  return (
    <DeleteButton
      onClick={() =>
        mutation.mutate(undefined, {
          onSuccess: () => {
            router.push('/datasets')
          },
        })
      }
    >
      Delete dataset
    </DeleteButton>
  )
}

const DatasetContext = createContext<Dataset>(null)

export function ViewDataset({ dataset }: { dataset: Dataset }) {
  const sections = ['manifest', 'description']
  const me = useEnhancedCurrentUser()

  return (
    <DatasetContext.Provider value={dataset}>
      <article className="pb-20 space-y-6">
        <section className="flex max-w-md space-x-4">
          <ChangeDatasetName />
        </section>
        <Section title="Dataset Tags">
          <div className="space-y-4">
            <TagList />
            <AddTag />
          </div>
        </Section>
        <Section title="Data">
          <Table headers={['File name', 'Syft id', 'dtype', 'Shape']}>
            {dataset.data
              ?.sort((a, b) => a?.name?.localeCompare(b?.name))
              .map((entry, index) => (
                <TableRow key={entry.id} darkBackground={index % 2}>
                  <TableData className="text-sm font-medium text-gray-700">
                    {entry.name}
                  </TableData>
                  <TableData>{entry.id}</TableData>
                  <TableData>{entry.dtype}</TableData>
                  <TableData>{entry.shape}</TableData>
                </TableRow>
              ))}
          </Table>
        </Section>
        {sections.map((section) => (
          <Section key={section} title={section}>
            <p className="text-sm whitespace-pre-wrap">{dataset[section]}</p>
          </Section>
        ))}
        {me?.permissions?.canUploadData && <DeleteDataset />}
      </article>
    </DatasetContext.Provider>
  )
}
