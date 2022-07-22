import { useMemo } from 'react'
import cn from 'classnames'
import Link from 'next/link'
import { dateFromNow, entityColors, localeSortByVariable } from '@/utils'
import { Badge, Spinner } from '@/components'
import type { Tensor, Dataset, Model } from '@/types/grid-types'

function prepareTensorsForAssetList(tensor: Tensor) {
  return {
    title: tensor.name,
    id: tensor.id,
    description: tensor.description || tensor.tags?.join(', ') || '',
    type: 'tensor',
    createdAt: 'Unknown',
    href: '/tensors',
  }
}

function prepareDatasetForAssetList(dataset: Dataset) {
  return {
    title: dataset.name,
    description: dataset.description,
    id: dataset.id,
    createdAt: dataset.createdAt ? dateFromNow(dataset.createdAt) : 'Unknown',
    type: 'dataset',
    href: '/datasets',
  }
}

function prepareModelForAssetList(model: Model) {
  return {
    title: model.name,
    description: model.description,
    id: model.id,
    createdAt: model?.createdAt ? dateFromNow(model.createdAt) : 'Unknown',
    type: 'model',
    href: '/models',
  }
}

function AssetListItem({ title, href, description, createdAt, type, id }) {
  return (
    <li key={id} className="px-4 py-4 col-span-full hover:bg-gray-100">
      <Link href={`${href}?v=${id}`}>
        <a>
          <div className="flex items-center justify-between space-x-4">
            <div className="flex items-center space-x-2">
              <span className="">{title || `Unknown ${type}`}</span>
              <Badge bgColor={entityColors[type]}>{type}</Badge>
            </div>
            <span className="text-sm text-gray-500">{createdAt}</span>
          </div>
          <div className="mt-1 text-sm text-gray-500 overflow-ellipsis line-clamp-3">
            {description}
          </div>
          <span className="text-xs text-gray-400">{id}</span>
        </a>
      </Link>
    </li>
  )
}

export interface AssetList {
  datasets: Dataset[]
  models: Model[]
  tensors: Tensor[]
}

export function LatestAssetsList({ datasets = [], models = [], tensors = [] }) {
  const assets = useMemo(
    () => [
      ...localeSortByVariable(
        datasets.map(prepareDatasetForAssetList).slice(0, 3),
        'createdAt'
      ),
      ...models.map(prepareModelForAssetList).slice(0, 3),
      ...tensors.map(prepareTensorsForAssetList).slice(0, 3),
    ],
    [datasets, models, tensors]
  )
  return (
    <ul className="grid grid-cols-1 divide-y divide-gray-300 md:grid-cols-8">
      {assets.map(AssetListItem)}
    </ul>
  )
}

export interface Card {
  link: string
  bgColor: string
  icon: React.ElementType
  main: string
  value: string | number
}

export function MiniCard({ link, bgColor, icon: Icon, main, value }: Card) {
  return (
    <li
      key={link}
      className={`relative col-span-1 flex shadow-sm rounded-md hover:ring-2 hover:ring-${bgColor}-500`}
    >
      <div
        className={cn(
          bgColor ? `bg-${bgColor}-600` : 'bg-cyan-600',
          'flex-shrink-0 flex items-center justify-center w-16 text-white text-sm font-medium rounded-l-md'
        )}
      >
        {Icon && <Icon className="w-8 h-8 text-white" aria-hidden />}
      </div>
      <div className="flex justify-between flex-1 truncate bg-white border-t border-b border-r border-gray-200 rounded-r-md">
        <div className="flex-1 px-4 py-2 text-sm truncate">
          <Link href={link}>
            <a className="text-gray-500 hover:text-gray-800">
              <span className="absolute inset-0" aria-hidden="true" />
              {main}
            </a>
          </Link>
          <p className="text-2xl font-semibold text-gray-900 truncate">
            {typeof value !== 'undefined' ? (
              value
            ) : (
              <Spinner className="w-3 h-3" />
            )}
          </p>
        </div>
      </div>
    </li>
  )
}

export function DashboardCards({ cards }: { cards: Card[] }) {
  return (
    <ul className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4">
      {cards.map(MiniCard)}
    </ul>
  )
}
