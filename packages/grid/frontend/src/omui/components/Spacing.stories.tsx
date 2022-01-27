import React from 'react'
import { Story, Meta } from '@storybook/react'
import theme from 'tailwindcss/defaultTheme'

const SpacingStory = args => {
  const color = args.color
  const spacing = ['px', 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 56, 64]

  return (
    <div>
      <h1>Spacing</h1>
      {spacing.map(size => (
        <div className="my-2 flex space-x-4 items-center">
          <span className="w-8">{size}</span>
          <div className={`w-${size} h-${size} bg-${color}-300 bg-opacity-50`} />
        </div>
      ))}
    </div>
  )
}

const BarSpacing = args => {
  const color = args.color
  const spacing = ['px', 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 56, 64]

  return (
    <div className="grid grid-cols-8 gap-4">
      <span>Name</span>
      <span>Space</span>
      <span className="col-span-6">Pixels</span>
      {spacing.map(size => (
        <>
          <span>{size}</span>
          <span>{theme.spacing[size]}</span>
          <span>
            {isNaN(size) ? theme.spacing[size] : `${parseFloat(theme.spacing[size]) * 16}px`}
          </span>
          <div className={`col-span-5 w-${size} h-4 bg-${color}-500 bg-opacity-50`} />
        </>
      ))}
    </div>
  )
}

export default {
  title: 'Atoms/Spacing',
  component: SpacingStory,
  argTypes: {
    color: {
      defaultValue: 'violet',
      options: [
        'magenta',
        'gray',
        'lime',
        'marigold',
        'orange',
        'cyan',
        'blue',
        'violet',
        'purple',
        'red',
      ].sort(),
      control: { type: 'select' },
    },
  },
} as Meta

export const SpacingBars = BarSpacing.bind({})

export const Spacing = SpacingStory.bind({})
