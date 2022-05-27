import React, { useState } from 'react'
import { Select } from './Select'
import type { SelectProps } from './Select'
import type { Story, Meta } from '@storybook/react'

export const Template: Story<SelectProps> = (args) => {
  const [selected, setSelected] = useState<string | number | null>()

  return (
    <div style={{ height: 300 }}>
      <Select
        {...args}
        value={selected}
        onChange={(value) => setSelected(value)}
        options={Array.from(Array(10).keys()).map((i) => ({
          value: i,
          label: `Element ${i}`,
        }))}
      />
    </div>
  )
}

export default {
  title: 'Components/Select',
  component: Select,
  parameters: {
    controls: {
      include: ['size', 'placeholder', 'disabled', 'error'],
    },
  },
} as Meta
