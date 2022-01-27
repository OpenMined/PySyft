import React from 'react'
import { Story, Meta } from '@storybook/react'

const Color = args => {
  const color = args.color
  const colorMaps = Array.from({ length: 9 }, (_, i) => 1000 - 100 * (10 - (i + 1)))

  colorMaps.unshift(50)
  return (
    <>
      <h1 className="text-2xl capitalize mb-4">{color}</h1>
      {colorMaps.map(v => (
        <div
          className={`bg-${color}-${v} h-16 w-32 flex text-${color}-${
            v > 300 ? 50 : 800
          } justify-center items-center`}
        >
          Text {v}
        </div>
      ))}
    </>
  )
}

export default {
  title: 'Atoms/Colors',
  component: Color,
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

export const Colors = Color.bind({})
