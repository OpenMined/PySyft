import { List } from '@/components'
import { dateFromNow } from '@/utils'
import type { Dataset } from '@/types/grid-types'

export function DatasetList({ datasets }: { datasets: Dataset[] }) {
  return (
    <List>
      {datasets.map(
        ({ id, name, createdAt, data, tags, description }: Dataset) => (
          <List.Item key={id} href={`/datasets?d=${id}`}>
            <div className="flex items-center justify-between truncate space-x-6">
              <p className="font-medium truncate">
                {name || 'Unnamed dataset'}
              </p>
              <div className="text-sm text-gray-500">
                {dateFromNow(createdAt)}
              </div>
            </div>
            <p className="text-sm text-gray-600">{data?.length} tensors</p>
            <p className="text-xs text-gray-500">{id}</p>
            <div className="text-sm text-gray-700">
              <p className="mt-4 overflow-ellipsis line-clamp-4 pr-12">
                {description}
              </p>
              <p className="mt-2 text-sm text-gray-500">{tags?.join(', ')}</p>
            </div>
          </List.Item>
        )
      )}
    </List>
  )
}
