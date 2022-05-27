import React from 'react'
import { Story, Meta } from '@storybook/react'

import { Tag, TagSizeProp } from './Tag'
import type { TagProps, TagVariantProp } from './Tag'

const RandomIcon = (props) => (
  <svg viewBox="0 -2 16 16" xmlns="http://www.w3.org/2000/svg" {...props}>
    <path
      fill="currentColor"
      d="M5.73047 10.7812C6.00391 11.0547 6.46875 11.0547 6.74219 10.7812L14.7812 2.74219C15.0547 2.46875 15.0547 2.00391 14.7812 1.73047L13.7969 0.746094C13.5234 0.472656 13.0859 0.472656 12.8125 0.746094L6.25 7.30859L3.16016 4.24609C2.88672 3.97266 2.44922 3.97266 2.17578 4.24609L1.19141 5.23047C0.917969 5.50391 0.917969 5.96875 1.19141 6.24219L5.73047 10.7812Z"
    />
  </svg>
)

export default {
  title: 'Components/Tag',
  component: Tag,
  argTypes: {
    onClick: {
      mapping: {
        true: () => console.log('click:omui-tag'),
        false: null,
      },
      defaultValue: false,
      control: 'boolean',
    },
  },
  parameters: {
    controls: {
      include: [
        'variant',
        'size',
        'disabled',
        'tagType',
        'className',
        'icon',
        'iconSide',
        'onClick',
      ],
    },
  },
} as Meta

const Template: Story<TagProps> = (args) => (
  <Tag {...args} icon={RandomIcon}>
    Text here
  </Tag>
)

const WithoutIconStory: Story<TagProps> = (args) => (
  <Tag {...args}>Text here</Tag>
)

const variants: TagVariantProp[] = ['gray', 'primary', 'quaternary', 'tertiary']
const tagSizes: TagSizeProp[] = ['sm', 'md', 'lg']

const rows: Array<Partial<TagProps>> = [
  { tagType: 'round' },
  { tagType: 'round', icon: RandomIcon, iconSide: 'right' },
  { tagType: 'round', icon: RandomIcon, iconSide: 'left' },
  { tagType: 'square' },
  { tagType: 'square', icon: RandomIcon, iconSide: 'left' },
  { tagType: 'square', icon: RandomIcon, iconSide: 'left' },
]

const AllTagsStory: Story<TagProps> = (args) => (
  <div className="space-y-8">
    {rows.map((props) => (
      <div className="space-y-2">
        {tagSizes.map((size) => (
          <div className="space-x-3">
            {variants.map((variant) => (
              <Tag {...args} {...props} size={size} variant={variant}>
                Text here
              </Tag>
            ))}
          </div>
        ))}
      </div>
    ))}
  </div>
)

export const DefaultWithIcon = Template.bind({})
DefaultWithIcon.argTypes = {
  iconSide: {
    defaultValue: 'left',
  },
}
DefaultWithIcon.parameters = {
  controls: {
    include: [
      'variant',
      'size',
      'disabled',
      'tagType',
      'className',
      'iconSide',
      'onClick',
    ],
  },
}

export const DefaultWithoutIcon = WithoutIconStory.bind({})
DefaultWithoutIcon.parameters = {
  controls: {
    include: ['variant', 'size', 'disabled', 'tagType', 'className', 'onClick'],
  },
}

export const AllTags = AllTagsStory.bind({})
AllTags.parameters = {
  controls: {
    include: ['className', 'onClick'],
  },
}
