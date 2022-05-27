import React from 'react'
import { Story, Meta } from '@storybook/react'
import { Card } from './Card'
import type { CardProps, CardSizeProp, CardVariantProp } from './Card'

export const Template: Story<CardProps> = (args) => (
  <Card {...args}>Content</Card>
)

const variants = [
  'coming',
  'default',
  'progress',
  'completed',
] as CardVariantProp[]
const size = ['S', 'M'].reverse() as CardSizeProp[]

export const AllCards: Story<Omit<CardProps, 'ref' & 'key'>> = (argTypes) => {
  return (
    <div className="space-y-12">
      {size.map((s: CardSizeProp) => (
        <div className="space-y-4">
          {variants.map((v: CardVariantProp) => (
            <div className="space-y-4">
              <Card size={s} variant={v} {...argTypes} />
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

AllCards.parameters = {
  controls: { exclude: ['variant', 'size', 'onClick'] },
}

export default {
  title: 'Components/Card',
  component: Card,
  argTypes: {
    title: {
      defaultValue: 'Foundations of Private Computation',
      type: 'string',
    },
    subTitle: {
      defaultValue: 'Free',
      type: 'string',
    },
    srcImage: {
      defaultValue:
        'https://images.unsplash.com/photo-1588892487050-67d92a5f4136',
      type: 'string',
    },
    progress: {
      defaultValue: 60,
      control: { type: 'range', min: 0, max: 100, step: 1 },
    },
    tags: {
      defaultValue: [{ name: 'Text here' }],
      type: [{ name: 'string' }],
    },
  },
} as Meta
