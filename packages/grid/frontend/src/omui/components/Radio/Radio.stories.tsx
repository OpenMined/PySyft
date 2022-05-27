import React, { useState } from 'react'
import { Radio } from './Radio'
import type { RadioProps } from './Radio'
import type { Story, Meta } from '@storybook/react'
import {
  RadioGroup as RadioGroupComponent,
  RadioGroupProps,
} from './RadioGroup'

const states: Partial<RadioProps>[] = [{}, { disabled: true }]
const cases: Partial<RadioProps>[] = [{}, { checked: true }]

const AllCases: Story<RadioProps> = () => {
  return (
    <div className="flex">
      <div className="flex space-x-6 p-4 rounded-md max-w-min border border-dashed border-purple-400">
        {cases.map((casesProp, casesIdx) => (
          <div className="space-y-3" key={casesIdx}>
            {states.map((statesProp, statesIdx) => (
              <Radio
                key={statesIdx}
                label="Label"
                {...statesProp}
                {...casesProp}
                onChange={() => void 0}
              />
            ))}
          </div>
        ))}
      </div>
      <div className="dark flex space-x-6 bg-gray-900 ml-8 p-4 rounded-md max-w-min border border-dashed border-purple-400">
        {cases.map((casesProp, casesIdx) => (
          <div className="space-y-3" key={casesIdx}>
            {states.map((statesProp, statesIdx) => (
              <Radio
                key={statesIdx}
                label="Label"
                {...statesProp}
                {...casesProp}
                onChange={() => void 0}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

export const Default: Story<RadioProps> = (args) => {
  const [checked, setChecked] = useState<boolean>(false)
  return (
    <Radio
      {...args}
      checked={checked}
      onChange={(e) => setChecked(e.target.checked)}
    />
  )
}

const GroupedValues: Story<RadioProps & RadioGroupProps> = (args) => {
  const [checked, setChecked] = useState<string>('foundations-of-private')
  return (
    <RadioGroupComponent
      name="which"
      value={checked}
      onChange={setChecked}
      inline={args.inline}
    >
      <Radio value="privacy-opportunity" label="Our Privacy Opportunity" />
      <Radio
        value="foundations-of-private"
        label="Foundations of Private Computation"
      />
      <Radio
        value="federated-learning"
        label="Federated Learning Across Enterprises"
      />
      <Radio value="federated-on-mobile" label="Federated Learning on Mobile" />
    </RadioGroupComponent>
  )
}

export default {
  title: 'Components/Radio',
  component: Radio,
  parameters: {
    controls: {
      include: ['label', 'inline'],
    },
  },
} as Meta

export const AllRadios = AllCases.bind({})

export const RadioGroup = GroupedValues.bind({})

RadioGroup.argTypes = {
  inline: {
    control: 'boolean',
  },
}
