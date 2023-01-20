import { Fragment } from 'react'
import cn from 'classnames'
import { Listbox } from '@headlessui/react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faCheck, faChevronDown } from '@fortawesome/free-solid-svg-icons'
import { Text } from '@/omui'

export default function Select({ options, value, onChange }) {
  return (
    <div className="relative">
      <Listbox value={value} onChange={onChange}>
        <Listbox.Button className="border space-x-8 flex justify-between items-center py-2 px-3 rounded active:text-gray-600 active:border-primary-500 focus:shadow-primary-focus text-gray-400">
          <Text>{value}</Text>
          <FontAwesomeIcon icon={faChevronDown} className="text-gray-600" />
        </Listbox.Button>
        <Listbox.Options className="absolute mt-2.5 max-w-max bg-primary-50 shadow-md py-2.5 space-y-1.5">
          {options.map((option) => (
            <Listbox.Option key={option.id} value={option.name} as={Fragment}>
              {({ active, selected }) => (
                <li
                  className={cn(
                    'px-2.5 cursor-pointer',
                    selected && 'bg-primary-500 text-white',
                    active && 'hover:font-bold'
                  )}
                >
                  <div className="relative">
                    {selected && (
                      <FontAwesomeIcon
                        icon={faCheck}
                        className="absolute inset-0 top-1"
                      />
                    )}
                    <Text className="pl-6 pr-10" as="p">
                      {option.name}
                    </Text>
                  </div>
                </li>
              )}
            </Listbox.Option>
          ))}
        </Listbox.Options>
      </Listbox>
    </div>
  )
}
