import React from 'react'
import { Divider } from './Divider'
import type { Story, Meta } from '@storybook/react'
import type { DividerProps, DividerColor } from './Divider'

export default {
  title: 'Atoms/Divider',
  component: Divider,
} as Meta

const Template: Story<DividerProps> = args => (
  <div style={{ height: 200 }}>
    <Divider {...args} />
  </div>
)

const AllColors: Story<DividerProps> = args => {
  const colors: DividerColor[] = ['dark', 'light', 'black']
  return (
    <div className={args.orientation === 'vertical' ? 'flex h-96 space-x-6' : 'w-96 space-y-6'}>
      {colors.map(color => (
        <Divider {...args} color={color} />
      ))}
    </div>
  )
}

export const Default = Template.bind({})

export const AllHorizontal = AllColors.bind({})
AllHorizontal.parameters = {
  controls: { disable: true },
}

export const AllVertical = AllColors.bind({})
AllVertical.parameters = {
  controls: { disable: true },
}
AllVertical.args = {
  orientation: 'vertical',
}
