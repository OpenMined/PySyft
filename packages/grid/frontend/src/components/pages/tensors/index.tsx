import { List } from '@/components'
import type { Tensor } from '@/types/grid-types'

export function TensorList({ tensors }: { tensors: Tensor[] }) {
  return (
    <List>
      {tensors.map(tensor => (
        <List.Item key={tensor.id}>
          <div className="px-4 py-5">
            <div className="flex items-center justify-between w-full">
              <p>Unnamed tensor</p>
              <p className="text-xs text-gray-500">{tensor.id}</p>
            </div>
            <div className="text-sm text-gray-500">
              <p>{tensor.description}</p>
              <p>{tensor.tags.join?.(' ')}</p>
            </div>
          </div>
        </List.Item>
      ))}
    </List>
  )
}
