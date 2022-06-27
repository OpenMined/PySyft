import { useEffect, useState } from 'react'
import { Search } from '@/components'
import type { Dispatch, SetStateAction } from 'react'

interface SearchProps<T> {
  data: T[]
  searchFields: string[]
  setData: Dispatch<SetStateAction<T[]>>
}

export function SearchBar<T>({ data, searchFields, setData }: SearchProps<T>) {
  const [search, setSearch] = useState<string>('')

  useEffect(() => {
    if (!search) {
      setData(null)
      return
    }

    if (Array.isArray(data)) {
      setData(
        data.filter((entry) => {
          const searchString = new RegExp(search, 'i')

          return searchFields.some((variable) => {
            const variableType = typeof entry?.[variable]

            if (variableType === 'string') {
              return entry[variable].search(searchString) !== -1
            }

            if (variableType === 'number') {
              return String(entry[variable]).search(searchString) !== -1
            }

            if (Array.isArray(entry?.[variable])) {
              return entry[variable].join(' ').search(searchString) !== -1
            }

            return false
          })
        })
      )
    }
  }, [data, searchFields, search, setData])

  return <Search setSearch={setSearch} />
}
