import { Fragment, useMemo, useState, useEffect } from 'react'
import cn from 'classnames'
import { ChevronDownIcon } from '@heroicons/react/outline'
import { Popover, Transition } from '@headlessui/react'
import { NormalButton } from '@/components'
import { gridPermissions } from '@/utils'
import type { Dispatch, SetStateAction } from 'react'

interface Filter {
  setSelectedFilters: Dispatch<SetStateAction<boolean[]>>
}

export function PermissionsFilter({ setSelectedFilters }: Filter) {
  const [selectedPermissions, setSelected] = useState<boolean[]>(
    Object.keys(gridPermissions).map(() => false)
  )
  const totalSelected = useMemo(
    () => selectedPermissions.filter(item => item).length,
    [selectedPermissions]
  )

  const onChange = (index: number) => {
    setSelected(prev => {
      const existing = [].concat(prev)
      existing[index] = !existing[index]
      return existing
    })
  }

  useEffect(() => {
    setSelectedFilters(selectedPermissions)
  }, [selectedPermissions, setSelectedFilters])

  return (
    <Popover className="relative">
      {({ open }) => (
        <>
          <Popover.Button as={Fragment}>
            <NormalButton
              className={cn(
                open ? 'text-gray-800' : 'text-gray-500',
                'group flex',
                'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'
              )}
            >
              <span className="hidden sm:inline-block">Permissions filter</span>
              <span className="sm:hidden">Permissions</span>
              {totalSelected > 0 && <span>&nbsp;({totalSelected})</span>}
              <ChevronDownIcon
                className={cn(
                  open ? 'text-gray-600' : 'text-gray-400',
                  'ml-2 h-5 w-5 group-hover:text-gray-500 transition ease-in-out duration-150'
                )}
                aria-hidden="true"
              />
            </NormalButton>
          </Popover.Button>
          <Transition
            show={open}
            as={Fragment}
            enter="transition ease-out duration-200"
            enterFrom="opacity-0 translate-y-1"
            enterTo="opacity-100 translate-y-0"
            leave="transition ease-in duration-150"
            leaveFrom="opacity-100 translate-y-0"
            leaveTo="opacity-0 translate-y-1"
          >
            <Popover.Panel static className="absolute z-10 mt-3">
              <div className="rounded-lg shadow-lg ring-1 max-w-xs w-screen lg:max-w-3xl ring-black ring-opacity-5 overflow-hidden">
                <div className="relative grid gap-6 bg-blueGray-50 px-5 py-6 lg:gap-8 lg:p-8 grid-cols-1 lg:grid-cols-2">
                  {Object.keys(gridPermissions).map((key, index) => {
                    const item = gridPermissions[key]
                    return (
                      <div className="flex" key={`filter-${key}`}>
                        <div className="flex h-6 items-center">
                          <input
                            type="checkbox"
                            defaultChecked={selectedPermissions[index]}
                            onChange={() => onChange(index)}
                            className="flex items-start rounded-md hover:bg-gray-50 transition ease-in-out duration-150"
                          />
                        </div>
                        <div className="ml-4">
                          <p className="text-base font-medium text-gray-900">{item.name}</p>
                          <p className="mt-1 text-sm text-gray-500">{item.description}</p>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </Popover.Panel>
          </Transition>
        </>
      )}
    </Popover>
  )
}
