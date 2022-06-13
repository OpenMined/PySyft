import { formatBytes } from '@/utils/common'
import dayjs from 'dayjs'

export function FileDetails({ file }) {
  if (!file) {
    return null
  }

  return (
    <section>
      <h3 className="font-medium">File details</h3>
      <div className="flex flex-col space-y-1 text-sm text-gray-600">
        <span>Name: {file.name}</span>
        <span>Size: {formatBytes(file.size) ?? 'Unknown'}</span>
        <span>
          Last modified:{' '}
          {file.lastModifiedDate?.toString() ?? file.lastModified
            ? dayjs(file.lastModified).toString()
            : undefined ?? 'Unknown'}
        </span>
      </div>
    </section>
  )
}
