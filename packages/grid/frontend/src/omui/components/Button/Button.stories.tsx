import React from 'react'
import { Story, Meta } from '@storybook/react'

import { Button, IconButton } from './Button'
import type { ButtonSize, ButtonVariant, ButtonProps, IconButtonVariant } from './Button'

export default {
  title: 'Components/Button',
  component: Button,
  parameters: {
    controls: {
      include: ['variant', 'size', 'children'],
    },
  },
  argTypes: {
    children: {
      name: 'children',
      type: { name: 'string' },
      defaultValue: 'Button',
    },
  },
} as Meta

export const Default: Story<ButtonProps> = args => <Button {...args} />

const sizes: ButtonSize[] = ['lg', 'md', 'sm', 'xs']
const variants: ButtonVariant[] = ['gray', 'primary', 'outline', 'ghost', 'link']
const iconVariants: IconButtonVariant[] = ['gray', 'primary', 'outline']
const states = [{ disabled: false }, { disabled: true }]

const AllCasesButton: Story<ButtonProps> = () => (
  <div className="flex space-x-8">
    {sizes.map(size => (
      <div className="flex flex-col space-y-4">
        {variants.map(variant => (
          <div className="flex space-x-4">
            {states.map(state => (
              <Button
                size={size}
                variant={variant}
                leftIcon={LightIcon}
                rightIcon={LightIcon}
                {...state}
              >
                Button
              </Button>
            ))}
          </div>
        ))}
      </div>
    ))}
  </div>
)

const AllCasesIconButton: Story = ({ rounded }: any) => (
  <div className="flex space-x-20">
    {sizes.map(size => (
      <div className="flex flex-col space-y-4">
        {iconVariants.map(variant => (
          <div className="flex space-x-4">
            {states.map(state => (
              <IconButton
                size={size}
                variant={variant}
                icon={SolidIcon}
                rounded={rounded}
                {...state}
              />
            ))}
          </div>
        ))}
      </div>
    ))}
  </div>
)

const LightIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path
      d="M10 0.3125C4.64844 0.3125 0.3125 4.64844 0.3125 10C0.3125 15.3516 4.64844 19.6875 10 19.6875C15.3516 19.6875 19.6875 15.3516 19.6875 10C19.6875 4.64844 15.3516 0.3125 10 0.3125ZM10 17.8125C5.66406 17.8125 2.1875 14.3359 2.1875 10C2.1875 5.70312 5.66406 2.1875 10 2.1875C14.2969 2.1875 17.8125 5.70312 17.8125 10C17.8125 14.3359 14.2969 17.8125 10 17.8125ZM6.875 9.375C7.53906 9.375 8.125 8.82812 8.125 8.125C8.125 7.46094 7.53906 6.875 6.875 6.875C6.17188 6.875 5.625 7.46094 5.625 8.125C5.625 8.82812 6.17188 9.375 6.875 9.375ZM13.125 9.375C13.7891 9.375 14.375 8.82812 14.375 8.125C14.375 7.46094 13.7891 6.875 13.125 6.875C12.4219 6.875 11.875 7.46094 11.875 8.125C11.875 8.82812 12.4219 9.375 13.125 9.375ZM13.2812 12.2266C12.4609 13.2031 11.25 13.75 10 13.75C8.71094 13.75 7.5 13.2031 6.71875 12.2266C6.36719 11.8359 5.78125 11.7969 5.39062 12.1094C5 12.4219 4.92188 13.0469 5.27344 13.4375C6.44531 14.8438 8.16406 15.625 10 15.625C11.7969 15.625 13.5156 14.8438 14.6875 13.4375C15.0391 13.0469 15 12.4219 14.5703 12.1094C14.1797 11.7969 13.5938 11.8359 13.2812 12.2266Z"
      fill="currentColor"
    />
  </svg>
)

const SolidIcon = () => (
  <svg role="img" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 496 512">
    <path
      fill="currentColor"
      d="M248 8C111 8 0 119 0 256s111 248 248 248 248-111 248-248S385 8 248 8zm80 168c17.7 0 32 14.3 32 32s-14.3 32-32 32-32-14.3-32-32 14.3-32 32-32zm-160 0c17.7 0 32 14.3 32 32s-14.3 32-32 32-32-14.3-32-32 14.3-32 32-32zm194.8 170.2C334.3 380.4 292.5 400 248 400s-86.3-19.6-114.8-53.8c-13.6-16.3 11-36.7 24.6-20.5 22.4 26.9 55.2 42.2 90.2 42.2s67.8-15.4 90.2-42.2c13.4-16.2 38.1 4.2 24.6 20.5z"
    />
  </svg>
)

export const AllCases: Story = () => (
  <div className="space-y-4">
    <div className="space-y-12">
      <div className="p-6 rounded-md border border-dashed border-purple-400 space-y-8">
        <AllCasesButton />
        <AllCasesIconButton />
        <AllCasesIconButton rounded />
      </div>
      <div className="dark bg-gray-900 p-6 rounded-md border border-dashed border-purple-400 space-y-8">
        <AllCasesButton />
        <AllCasesIconButton />
        <AllCasesIconButton rounded />
      </div>
    </div>
  </div>
)
