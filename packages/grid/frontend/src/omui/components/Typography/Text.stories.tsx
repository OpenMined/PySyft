import React from 'react'
import { Story, Meta } from '@storybook/react'

import { Text, H1, H2, H3, H4, H5, H6, TextSizeProp, TextProps } from './Text'

export default {
  title: 'Atoms/Type',
  component: Text,
} as Meta

const types = ['xs', 'sm', 'md', 'lg', 'xl', '2xl', '3xl', '4xl', '5xl', '6xl'] as TextSizeProp[]

const Template: Story<TextProps> = args => (
  <>
    {types.map(type => (
      <Text {...args} size={type}>
        {type}
      </Text>
    ))}
  </>
)

const HeadingStory: Story = () => {
  return (
    <>
      <H1>Heading 1</H1>
      <H2>Heading 2</H2>
      <H3>Heading 3</H3>
      <H4>Heading 4</H4>
      <H5>Heading 5</H5>
      <H6>Heading 6</H6>
    </>
  )
}

export const Default = Template.bind({})

export const Headings = HeadingStory.bind({})
