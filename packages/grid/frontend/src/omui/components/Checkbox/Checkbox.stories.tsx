import React, { useState } from 'react'
import { Checkbox } from './Checkbox'
import type { CheckboxProps } from './Checkbox'
import type { Story, Meta } from '@storybook/react'

const types: Partial<CheckboxProps>[] = [{}, { label: 'Label' }]
const states: Partial<CheckboxProps>[] = [{}, { disabled: true }]
const cases: Partial<CheckboxProps>[] = [
  {},
  { checked: true },
  { indeterminate: true },
]

const AllCases: Story<CheckboxProps> = () => {
  const [values, setValues] = useState<
    { id: string | number; checked: boolean }[]
  >(Array.from(Array(10).keys()).map((val) => ({ id: val, checked: false })))
  const allChecked = values.every((v) => v.checked)
  const someChecked = values.some((v) => v.checked) && !allChecked

  function handleField(e: any, item: any) {
    setValues((old) =>
      old.map((i) =>
        i.id === item.id ? { ...item, checked: e.target.checked } : i
      )
    )
  }
  function handleToggleAll() {
    setValues((old) => old.map((i) => ({ ...i, checked: !allChecked })))
  }

  return (
    <div className="space-y-10">
      <div className="flex space-x-6">
        {types.map((typesProps, typeIdx) => (
          <Checkbox key={typeIdx} {...typesProps} onChange={() => void 0} />
        ))}
      </div>
      <div className="flex">
        <div className="flex space-x-12 p-4 rounded-md max-w-min border border-dashed border-purple-400">
          {states.map((stateProps, stateIdx) => (
            <div className="space-y-2" key={stateIdx}>
              {cases.map((props, idx) => (
                <Checkbox
                  key={idx}
                  {...stateProps}
                  {...props}
                  onChange={() => void 0}
                  label="Label"
                />
              ))}
            </div>
          ))}
        </div>
        <div className="dark flex space-x-12 bg-gray-900 ml-8 p-4 rounded-md max-w-min border border-dashed border-purple-400">
          {states.map((stateProps, stateIdx) => (
            <div className="space-y-2" key={stateIdx}>
              {cases.map((props, idx) => (
                <Checkbox
                  key={idx}
                  {...stateProps}
                  {...props}
                  onChange={() => void 0}
                  label="Label"
                />
              ))}
            </div>
          ))}
        </div>
      </div>
      <div className="space-y-1">
        <Checkbox
          indeterminate={someChecked}
          checked={allChecked}
          onChange={handleToggleAll}
          label="All fields"
        />
        {values.map((item) => (
          <div className="pl-4 space-y-2" key={item.id}>
            <Checkbox
              checked={item.checked}
              onChange={(e) => handleField(e, item)}
              label="Label"
            />
          </div>
        ))}
      </div>
    </div>
  )
}

export const Template: Story<CheckboxProps> = (args) => {
  const [checked, setChecked] = useState<boolean>(false)
  return (
    <>
      <Checkbox
        {...args}
        checked={checked}
        onChange={(e) => setChecked(e.target.checked)}
      />
      <Checkbox
        {...args}
        indeterminate={checked}
        onChange={(e) => setChecked(e.target.checked)}
      />
    </>
  )
}

export default {
  title: 'Components/Checkbox',
  component: Checkbox,
  parameters: {
    controls: {
      include: ['label', 'disabled'],
    },
  },
} as Meta

export const AllCheckboxes = AllCases.bind({})
