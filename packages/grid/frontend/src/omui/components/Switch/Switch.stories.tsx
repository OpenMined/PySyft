import React, { useState } from 'react'
import { Switch } from './Switch'
import type { SwitchProps, SwitchSizeProp } from './Switch'
import type { Story, Meta } from '@storybook/react'

const sizes: { size: SwitchSizeProp }[] = [{ size: 'sm' }, { size: 'md' }, { size: 'lg' }]
const states: Partial<SwitchProps>[] = [{}, { disabled: true }]
const cases: Partial<SwitchProps>[] = [{}, { checked: true }]

const AllCases: Story<SwitchProps> = () => {
  return (
    <div className="flex space-x-6">
      {states.map((statesProp, statesIdx) => (
        <div className="space-y-3" key={statesIdx}>
          {sizes.map((sizesProp, sizesIdx) => (
            <div className="flex space-x-3" key={sizesIdx}>
              {cases.map((casesProps, typeIdx) => (
                <Switch
                  key={typeIdx}
                  {...casesProps}
                  {...statesProp}
                  {...sizesProp}
                  onChange={() => void 0}
                />
              ))}
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

export const Template: Story<SwitchProps> = args => {
  const [checked, setChecked] = useState<boolean>(false)
  return <Switch {...args} checked={checked} onChange={value => setChecked(value)} />
}

export default {
  title: 'Components/Switch',
  component: Switch,
  parameters: {
    controls: {
      include: ['size', 'disabled'],
    },
  },
} as Meta

export const AllSwitches = AllCases.bind({})
