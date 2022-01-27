import { Fragment, useState } from 'react'
import { useForm } from 'react-hook-form'
import { XCircleIcon } from '@heroicons/react/solid'
import { Disclosure } from '@headlessui/react'
import { NormalButton, Input, Tag, Accordion } from '@/components'
import { useModels } from '@/lib/data'
import type { ChangeEventHandler } from 'react'
import type { Model } from '@/types/grid-types'

function TagsShowcase({
  id,
  tags,
  onRemove,
}: {
  id: string | number
  tags: string[]
  onRemove: (e: string) => void
}) {
  return (
    <div className="space-y-1">
      <p className="ml-1 text-sm font-medium text-gray-700">Tags</p>
      <div className="flex space-x-4">
        {tags.length === 0 && <p>This model has no tags.</p>}
        {tags.map(tag => (
          <div key={`${id}-${tag}`} className="flex">
            <Tag className="rounded-r-none bg-pink-50">{tag}</Tag>
            {tags.length > 1 && (
              <button
                className="flex items-center px-2 text-white bg-red-400 cursor-pointer rounded-r-md hover:bg-red-800"
                onClick={() => onRemove(tag)}
              >
                <XCircleIcon className="w-4 h-4" />
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function AddNewTag({ onAdd }) {
  const [tag, setTag] = useState('')
  const handleChange: ChangeEventHandler<HTMLInputElement> = event => {
    setTag(event.target.value)
  }

  const handleAdd = () => {
    if (tag) {
      onAdd(tag)
      setTag('')
    }
  }

  return (
    <div className="space-y-1">
      <label htmlFor="add" className="ml-1 text-sm font-medium text-gray-700">
        Add a tag
      </label>
      <div className="flex space-x-4">
        <div className="flex">
          <Input
            pre="#"
            id="add-tag"
            placeholder="tag name"
            className="rounded-l-none"
            onChange={handleChange}
            name="add"
            value={tag}
          />
        </div>
        <div className="mt-auto">
          <NormalButton className="hover:bg-trueGray-300" type="button" onClick={handleAdd}>
            Add
          </NormalButton>
        </div>
      </div>
    </div>
  )
}

function ModelEditFormButtons({ onCancel }) {
  return (
    <div className="w-full space-x-6 text-right">
      <NormalButton className="bg-blue-500 hover:bg-blue-800 text-gray-50">Save</NormalButton>
      <Disclosure.Button as={Fragment}>
        <NormalButton type="button" className="hover:bg-gray-300" onClick={onCancel}>
          Cancel
        </NormalButton>
      </Disclosure.Button>
    </div>
  )
}

type ModelEditForm = Pick<Model, 'name' | 'description'>

function EditPanel(model: Model) {
  const { update } = useModels()
  const mutation = update(model.id)
  const [tags, changeTags] = useState(model.tags)
  const { handleSubmit, register, reset } = useForm<ModelEditForm>()

  const onCancel = () => {
    reset()
    changeTags(() => [...model.tags])
  }

  const onSubmit = (values: ModelEditForm) => {
    mutation.mutate({ ...values, tags })
  }

  return (
    <div className="px-4 py-6 space-y-6 text-sm border-t border-gray-200 bg-blueGray-100">
      <fieldset disabled={mutation.isLoading}>
        <form className="space-y-6" onSubmit={handleSubmit(onSubmit)}>
          {['name', 'description'].map(type => (
            <Input
              key={`${model.id}-${type}`}
              id={`${model.id}-${type}`}
              name={type}
              label={type}
              defaultValue={model[type]}
              placeholder={`Add a model ${type}`}
              ref={register}
            />
          ))}
          <TagsShowcase
            tags={tags}
            id={model.id}
            onRemove={tag => {
              changeTags(tags => {
                const index = tags.indexOf(tag)
                const copy = [].concat(tags)
                copy.splice(index, 1)
                return copy
              })
            }}
          />
          <AddNewTag
            onAdd={newTag => {
              changeTags(tags => [...tags, `#${newTag}`])
            }}
          />
          <ModelEditFormButtons onCancel={onCancel} />
        </form>
      </fieldset>
    </div>
  )
}

export function ModelList({ models }: { models: Model[] }) {
  return (
    <Accordion>
      {models.map(model => (
        <Accordion.Item key={model.id}>
          <div className="w-full">
            <div className="flex justify-between truncate">
              <p className="font-medium truncate">{model.name || 'Unnamed model'}</p>
              <p className="text-xs text-gray-500">{model.id}</p>
            </div>
            <div className="space-y-2 text-sm text-gray-700">
              <p className="pr-12 mt-4 text-sm text-gray-700 overflow-ellipsis line-clamp-4">
                {model.description}
              </p>
              <div className="flex flex-wrap">
                {model.tags.map(tag => (
                  <Tag key={tag} className="mb-2 mr-2">
                    {tag}
                  </Tag>
                ))}
              </div>
            </div>
          </div>
          <EditPanel {...model} />
        </Accordion.Item>
      ))}
    </Accordion>
  )
}
