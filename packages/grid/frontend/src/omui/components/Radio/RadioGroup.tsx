import React, { createContext, useContext, useMemo } from 'react'
import type { ContextType } from 'react'

type RadioGroupContextValues = {
  name: string
  onChange: (value: string) => void
  value?: string
} | null
const RadioGroupContext = createContext<RadioGroupContextValues>(null)
RadioGroupContext.displayName = 'RadioGroupContext'

function useRadioGroupContext() {
  const context = useContext(RadioGroupContext)
  return context
}

export type RadioGroupProps = {
  /**
   * The name attribute is forwarded to every `Radio` children element.
   */
  name: string
  /**
   * The value of the Radio to be `checked`.
   */
  value?: string
  /**
   * Function called once a Radio changes.
   * @param value the value of the checked radio
   */
  onChange: (value: string) => void
  /**
   * Define the layout of the Radio Group as horizontal.
   */
  inline?: boolean
  children: React.ReactNode
}

function RadioGroup({
  name,
  value,
  onChange,
  inline,
  children,
}: RadioGroupProps) {
  const radioGroupApi = useMemo<ContextType<typeof RadioGroupContext>>(
    () => ({ name, value, onChange }),
    [name, value, onChange]
  )
  return (
    <RadioGroupContext.Provider value={radioGroupApi}>
      <div
        role="radiogroup"
        className={inline ? 'inline-flex space-x-6' : undefined}
      >
        {children}
      </div>
    </RadioGroupContext.Provider>
  )
}

export { RadioGroup, useRadioGroupContext }
