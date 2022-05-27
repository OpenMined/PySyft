import React from 'react'
import { Story, Meta } from '@storybook/react'

import { Icon, IconContainerProp } from './Icon'
import type { IconProps, IconVariantProp, IconSizeProp } from './Icon'

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

export const Template: Story<IconProps> = (args) => (
  <Icon {...args} icon={RandomIcon} />
)

const size = ['xs', 'sm', 'md', 'lg', 'xl'].reverse() as IconSizeProp[]
const container: IconContainerProp[] = ['square', 'round']

export const AllIcons: Story<IconProps> = (_, { argTypes }) => {
  const { variant } = argTypes
  return (
    <div className="space-y-12">
      {container.map((c: IconContainerProp) => (
        <div className="space-y-4">
          {variant.options.map((v: IconVariantProp) => (
            <div className="flex items-center space-x-4">
              {size.map((s: IconSizeProp) => (
                <Icon container={c} size={s} variant={v} icon={RandomIcon} />
              ))}
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}
export default {
  title: 'Atoms/Icon',
  component: Icon,
  argTypes: {
    icon: { control: false },
    ref: { table: { disable: true } },
    key: { table: { disable: true } },
  },
} as Meta

AllIcons.parameters = {
  controls: {
    disabled: true,
  },
}
