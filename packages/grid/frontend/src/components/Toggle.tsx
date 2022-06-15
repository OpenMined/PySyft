import { useState } from 'react'
import { Switch } from '@headlessui/react'

export function Toggle({ alt = '', initialEnabled = false, onChange }) {
  const [enabled, setEnabled] = useState<boolean>(initialEnabled)

  const handleChange = (isEnabled: boolean) => {
    setEnabled(isEnabled)
    onChange(isEnabled)
  }

  return (
    <div>
      <Switch
        checked={enabled}
        onChange={handleChange}
        className={`${enabled ? 'bg-green-500' : 'bg-gray-300'}
          relative inline-flex flex-shrink-0 h-6 w-12 border-2 border-transparent rounded-full cursor-pointer transition-colors ease-in-out duration-200 focus:outline-none focus-visible:ring-2  focus-visible:ring-white focus-visible:ring-opacity-75`}
      >
        <span className="sr-only">{alt}</span>
        <span
          aria-hidden="true"
          className={`${enabled ? 'translate-x-6' : 'translate-x-0'}
            pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow-lg transform ring-0 transition ease-in-out duration-200`}
        />
      </Switch>
    </div>
  )
}
