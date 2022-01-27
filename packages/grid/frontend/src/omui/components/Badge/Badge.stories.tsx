import React from 'react'
import { Story, Meta } from '@storybook/react'
import { Badge } from './Badge'
import type { BadgeProps, BadgeTypeProp, BadgeVariantProp } from './Badge'

export default {
  title: 'Components/Badges',
  component: Badge,
} as Meta

const Template: Story<BadgeProps> = args => <Badge {...args}>{args.variant}</Badge>

const types: BadgeTypeProp[] = ['outline', 'subtle', 'solid']
const variants: BadgeVariantProp[] = [
  'gray',
  'primary',
  'tertiary',
  'quaternary',
  'danger',
  'success',
]

const AllBadges = () => (
  <div className="space-y-4">
    {variants.map(variant => (
      <div className="flex space-x-2">
        {types.map(type => (
          <div className="capitalize">
            <Badge type={type} variant={variant}>
              {variant}
            </Badge>
          </div>
        ))}
      </div>
    ))}
  </div>
)

export const Default = Template.bind({})

export const AllCasesLight = AllBadges.bind({})
