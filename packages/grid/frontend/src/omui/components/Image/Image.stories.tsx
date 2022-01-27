import React from 'react'
import { Image } from './Image'
import type { ImageProps } from './Image'
import type { Story, Meta } from '@storybook/react'

const Template: Story<ImageProps> = args => {
  return (
    <div className="w-2/4">
      <Image {...args} />
    </div>
  )
}

const MultiTemplate: Story<ImageProps> = args => {
  return (
    <div className="w-1/4 space-y-6">
      <div className="w-1/4">
        <Image {...args} />
      </div>
      <div className="w-2/4">
        <Image {...args} />
      </div>
      <div className="w-3/4">
        <Image {...args} />
      </div>
      <div className="w-full">
        <Image {...args} />
      </div>
    </div>
  )
}

export default {
  title: 'Atoms/Images',
  component: Image,
  parameters: {
    controls: {
      include: ['alt', 'ratio', 'orientation'],
    },
  },
} as Meta

export const emptyImage = Template.bind({})

export const imageOfAFox = Template.bind({})
imageOfAFox.args = {
  ...emptyImage.args,
  alt: 'A beautiful fox',
  src: 'https://images.unsplash.com/photo-1623288749528-e40a033da0f7',
}

export const imageOfAFoxInDifferentlySizedContainers = MultiTemplate.bind({})
imageOfAFoxInDifferentlySizedContainers.args = {
  ...emptyImage.args,
  alt: 'A beautiful fox',
  src: 'https://images.unsplash.com/photo-1623288749528-e40a033da0f7',
}
