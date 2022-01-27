import React, { useRef, useState } from 'react'
import { FormControl } from './FormControl'
import type { FormControlProps } from './FormControl'
import type { Story, Meta } from '@storybook/react'
import { Input } from '../Input/Input'

const cases: Omit<FormControlProps, 'children'>[] = [
  { id: 'default', label: 'Label', helperText: 'Hint Text' },
  { id: 'disabled', label: 'Label', helperText: 'Hint Text', disabled: true },
  { id: 'error', label: 'Label', helperText: 'Hint Text', error: true },
]

export const Default: Story<FormControlProps> = args => {
  return (
    <FormControl {...args} id="default">
      <Input placeholder="Text Here" />
    </FormControl>
  )
}

export const AllCases: Story<FormControlProps> = () => {
  return (
    <div className="flex space-x-16">
      <div className="bg-white p-6 w-full max-w-sm rounded-md border border-dashed border-purple-400 space-y-16">
        {cases.map(formCase => (
          <FormControl {...formCase} required key={formCase.id}>
            <Input placeholder="Text Here" />
          </FormControl>
        ))}
      </div>
      <div className="dark w-full max-w-sm bg-gray-900 p-6 rounded-md border border-dashed border-purple-400 space-y-16">
        {cases.map(formCase => (
          <FormControl
            {...formCase}
            id={`dark-${formCase.id}`}
            required
            key={`dark-${formCase.id}`}
          >
            <Input placeholder="Text Here" />
          </FormControl>
        ))}
      </div>
    </div>
  )
}
export default {
  title: 'Components/FormControl',
  component: FormControl,
  argTypes: {
    label: {
      defaultValue: 'Label',
      control: { type: 'text' },
    },
    helperText: {
      defaultValue: 'Hint text',
      control: { type: 'text' },
    },
  },
} as Meta
