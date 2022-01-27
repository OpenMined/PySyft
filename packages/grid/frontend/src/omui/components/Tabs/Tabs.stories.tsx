import React, { useState } from 'react'
import { Tabs, TabsAlignProp, TabsVariantProp } from './Tabs'
import type { TabsProps } from './Tabs'
import type { Story, Meta } from '@storybook/react'

const tabsList = [
  { title: 'Tab Name', id: 1 },
  { title: 'Tab Name', id: 2 },
  { title: 'Tab Name', id: 3 },
  { title: 'Tab Name', id: 4 },
  { title: 'Tab Name', id: 5, disabled: true },
]

export const Default: Story<TabsProps> = args => {
  const [selected, setSelected] = useState<string | number>(3)

  return (
    <Tabs {...args} active={selected} tabsList={tabsList} onChange={value => setSelected(value)}>
      <h1>{selected}</h1>
    </Tabs>
  )
}

const variants: TabsVariantProp[] = ['outline', 'underline']
const alignments: TabsAlignProp[] = ['left', 'right', 'auto']

export const AllTabs: Story = () => {
  const [selected, setSelected] = useState<string | number>(3)

  return (
    <div className="space-y-28">
      {variants.map((variantProp, variantIdx) => (
        <div key={variantIdx} className="space-y-4">
          {alignments.map((alignProp, alignIdx) => (
            <Tabs
              key={alignIdx}
              align={alignProp}
              variant={variantProp}
              active={selected}
              tabsList={tabsList}
              onChange={setSelected}
            />
          ))}
        </div>
      ))}
    </div>
  )
}

export default {
  title: 'Components/Tabs',
  component: Tabs,
  parameters: {
    controls: {
      include: ['align', 'variant', 'size'],
    },
  },
} as Meta

AllTabs.argTypes = {
  controls: {
    include: [],
  },
}
