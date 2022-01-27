import React from 'react'
import { Story, Meta } from '@storybook/react'
import { Avatar } from './Avatar'

const AvatarStory = args => {
  return (
    <Avatar
      src="https://avataaars.io/?avatarStyle=Circle&topType=LongHairStraight&accessoriesType=Blank&hairColor=BrownDark&facialHairType=Blank&clotheType=BlazerShirt&eyeType=Default&eyebrowType=Default&mouthType=Default&skinColor=Light"
      size={args.size}
      show={args.show}
    />
  )
}

export default {
  title: 'Atoms/Avatar',
  component: AvatarStory,
  argTypes: {
    size: {
      defaultValue: 'md',
      options: ['sm', 'md', 'lg', 'xl', '2xl', '3xl'],
      control: { type: 'select' },
    },
    show: {
      control: { type: 'boolean' },
    },
  },
} as Meta

export const Avatars = AvatarStory.bind({})
