import React from 'react'
import { Story, Meta } from '@storybook/react'

import {
  List,
  ListAvatarItem,
  ListContainedItem,
  ListIconItem,
  ListItem,
  ListItemContent,
  ListProgressItem,
  OrderedList,
  UnorderedList,
  OrderedListItem,
  UnorderedListItem,
} from './List'
import type {
  ListContainedProps,
  ListAvatarItemProps,
  ListIconProps,
  ListProps,
} from './List'

export default {
  title: 'Components/List',
  component: List,
  parameters: {
    controls: {
      include: ['children', 'size'],
    },
  },
  argTypes: {
    children: {
      name: 'children',
      defaultValue: 'Text Here',
      control: { type: 'text' },
    },
  },
} as Meta

export const DefaultList: Story<ListProps> = (args) => (
  <List size={args.size}>
    {Array.from(Array(10).keys()).map((_, index) => (
      <ListItem key={`def-${index}`}>{args.children}</ListItem>
    ))}
  </List>
)

export const Ordered: Story<ListProps> = (args) => (
  <OrderedList size={args.size}>
    {Array.from(Array(10).keys()).map((_, index) => (
      <OrderedListItem key={`oli-${index}`}>{args.children}</OrderedListItem>
    ))}
  </OrderedList>
)

export const Unordered: Story<ListProps> = (args) => (
  <UnorderedList size={args.size}>
    {Array.from(Array(10).keys()).map((_, index) => (
      <UnorderedListItem key={`uli-${index}`}>
        {args.children}
      </UnorderedListItem>
    ))}
  </UnorderedList>
)

export const Avatar: Story<ListAvatarItemProps & ListProps> = (args) => (
  <List size={args.size}>
    {Array.from(Array(10).keys()).map((_, index) => (
      <ListAvatarItem
        key={`a-${index}`}
        src="https://images.unsplash.com/photo-1623288749528-e40a033da0f7"
      >
        {args.children}
      </ListAvatarItem>
    ))}
  </List>
)

export const AvatarWithListContent: Story<ListAvatarItemProps & ListProps> = (
  args
) => (
  <List size={args.size}>
    {Array.from(Array(10).keys()).map((_, index) => (
      <ListAvatarItem
        key={`avlc-${index}`}
        src="https://images.unsplash.com/photo-1623288749528-e40a033da0f7"
      >
        <ListItemContent
          label={`${args.label}`}
          description={`${args.description}`}
        />
      </ListAvatarItem>
    ))}
  </List>
)

AvatarWithListContent.argTypes = {
  label: {
    name: 'label',
    defaultValue: 'Text Here',
    control: { type: 'text' },
  },
  description: {
    name: 'description',
    defaultValue: 'Description Here',
    control: { type: 'text' },
  },
}

AvatarWithListContent.parameters = {
  controls: {
    include: ['size', 'label', 'description'],
    exclude: ['children'],
  },
}

const RandomIcon = ({ className }: { className: string }) => (
  <svg
    className={className}
    role="img"
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 496 512"
  >
    <path
      fill="currentColor"
      d="M248 8C111 8 0 119 0 256s111 248 248 248 248-111 248-248S385 8 248 8zm80 168c17.7 0 32 14.3 32 32s-14.3 32-32 32-32-14.3-32-32 14.3-32 32-32zm-160 0c17.7 0 32 14.3 32 32s-14.3 32-32 32-32-14.3-32-32 14.3-32 32-32zm194.8 170.2C334.3 380.4 292.5 400 248 400s-86.3-19.6-114.8-53.8c-13.6-16.3 11-36.7 24.6-20.5 22.4 26.9 55.2 42.2 90.2 42.2s67.8-15.4 90.2-42.2c13.4-16.2 38.1 4.2 24.6 20.5z"
    />
  </svg>
)

export const Icon: Story<ListIconProps & ListProps> = (args) => (
  <List size={args.size}>
    {Array.from(Array(10).keys()).map((_, index) => (
      <ListIconItem icon={RandomIcon} key={`ic-${index}`}>
        {args.children}
      </ListIconItem>
    ))}
  </List>
)

export const Progress: Story<ListProps> = (args) => (
  <List size={args.size}>
    {Array.from(Array(10).keys()).map((_, index) => (
      <ListProgressItem key={`p-${index}`}>{args.children}</ListProgressItem>
    ))}
  </List>
)

export const Contained: Story<ListContainedProps & ListProps> = (args) => (
  <List size={args.size}>
    {Array.from(Array(10).keys()).map((_, index) => (
      <ListContainedItem containedValue={index + 1} key={`con-${index}`}>
        {args.children}
      </ListContainedItem>
    ))}
  </List>
)
