import React from 'react'
import { Story, Meta } from '@storybook/react'

import { Input, InputProps } from './Input'

export default {
  title: 'Components/Input',
  component: Input,
  argTypes: {
    addonLeft: {
      mapping: {
        true: (
          <svg
            width="16"
            height="17"
            fill="currentColor"
            viewBox="0 0 16 17"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M8 0.75C3.71875 0.75 0.25 4.21875 0.25 8.5C0.25 12.7812 3.71875 16.25 8 16.25C12.2812 16.25 15.75 12.7812 15.75 8.5C15.75 4.21875 12.2812 0.75 8 0.75ZM8 14.75C4.53125 14.75 1.75 11.9688 1.75 8.5C1.75 5.0625 4.53125 2.25 8 2.25C11.4375 2.25 14.25 5.0625 14.25 8.5C14.25 11.9688 11.4375 14.75 8 14.75ZM5.5 8C6.03125 8 6.5 7.5625 6.5 7C6.5 6.46875 6.03125 6 5.5 6C4.9375 6 4.5 6.46875 4.5 7C4.5 7.5625 4.9375 8 5.5 8ZM10.5 8C11.0312 8 11.5 7.5625 11.5 7C11.5 6.46875 11.0312 6 10.5 6C9.9375 6 9.5 6.46875 9.5 7C9.5 7.5625 9.9375 8 10.5 8ZM10.625 10.2812C9.96875 11.0625 9 11.5 8 11.5C6.96875 11.5 6 11.0625 5.375 10.2812C5.09375 9.96875 4.625 9.9375 4.3125 10.1875C4 10.4375 3.9375 10.9375 4.21875 11.25C5.15625 12.375 6.53125 13 8 13C9.4375 13 10.8125 12.375 11.75 11.25C12.0312 10.9375 12 10.4375 11.6562 10.1875C11.3438 9.9375 10.875 9.96875 10.625 10.2812Z"
              fill="currentColor"
            />
          </svg>
        ),
        false: null,
      },
      defaultValue: false,
      control: 'boolean',
    },
    addonRight: {
      mapping: {
        true: <div>Addon Right</div>,
        false: null,
      },
      defaultValue: false,
      control: 'boolean',
    },
  },
} as Meta

const Template: Story<InputProps> = args => <Input {...args} />

export const Default = Template.bind({})
