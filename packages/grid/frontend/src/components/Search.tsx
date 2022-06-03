import { SearchIcon } from '@heroicons/react/outline'
import type { ChangeEventHandler } from 'react'

export function Search({ setSearch }) {
  const handleChange: ChangeEventHandler<HTMLInputElement> = (event) =>
    setSearch(event.target.value)

  return (
    <div className="w-full">
      <label htmlFor="search" className="sr-only">
        Search in PyGrid
      </label>
      <div className="relative rounded-md shadow-sm">
        <div
          className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"
          aria-hidden="true"
        >
          <SearchIcon
            className="mr-3 h-4 w-4 text-gray-400 group-hover:text-gray-600"
            aria-hidden="true"
          />
        </div>
        <input
          type="text"
          name="search"
          id="search"
          className="focus:ring-cyan-500 focus:border-cyan-500 block w-full pl-9 sm:text-sm border-gray-300 rounded-md"
          placeholder="Search"
          onChange={handleChange}
        />
      </div>
    </div>
  )
}
