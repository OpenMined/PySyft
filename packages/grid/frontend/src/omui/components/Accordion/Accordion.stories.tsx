import React, { useState } from 'react'
import {
  Accordion,
  AccordionButton,
  AccordionItem,
  AccordionPanel,
} from './Accordion'
import type { AccordionProps } from './Accordion'
import type { Story, Meta } from '@storybook/react'
import { Image } from '../Image/Image'

export const Default: Story<AccordionProps> = (args) => {
  return (
    <Accordion className="max-w-xs" {...args}>
      <AccordionItem>
        <h2>
          <AccordionButton>Text here</AccordionButton>
        </h2>

        <AccordionPanel>
          <p className="mb-4">
            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
            eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim
            ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut
            aliquip ex ea commodo consequat.
          </p>
          <div className="flex flex-col space-y-4">
            <input type="text" />
            <input type="text" />
            <input type="text" />
            <input type="text" />
          </div>
        </AccordionPanel>
      </AccordionItem>

      <AccordionItem>
        <h2>
          <AccordionButton>Text Here</AccordionButton>
        </h2>
        <AccordionPanel>
          Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
          eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad
          minim veniam, quis nostrud exercitation ullamco laboris nisi ut
          aliquip ex ea commodo consequat.
        </AccordionPanel>
      </AccordionItem>

      <AccordionItem disabled>
        <h2>
          <AccordionButton>Text Here</AccordionButton>
        </h2>
        <AccordionPanel>Disabled panel</AccordionPanel>
      </AccordionItem>

      <AccordionItem>
        <h2>
          <AccordionButton>Text Here</AccordionButton>
        </h2>
        <AccordionPanel>
          <p className="mb-4">
            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
            eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim
            ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut
            aliquip ex ea commodo consequat.
          </p>

          <div className="flex space-x-4">
            <button type="button">Button</button>
            <button type="button">Button</button>
          </div>
        </AccordionPanel>
      </AccordionItem>
    </Accordion>
  )
}

const Example: Story<AccordionProps> = (args) => {
  return (
    <div className="flex space-x-8">
      <Accordion className="max-w-xs" {...args}>
        <AccordionItem>
          <h2>
            <AccordionButton>Text here</AccordionButton>
          </h2>

          <AccordionPanel>
            <p className="mb-4">
              Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
              eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut
              enim ad minim veniam, quis nostrud exercitation ullamco laboris
              nisi ut aliquip ex ea commodo consequat.
            </p>
            <div className="flex flex-col space-y-4">
              <input type="text" />
              <input type="text" />
              <input type="text" />
              <input type="text" />
            </div>
          </AccordionPanel>
        </AccordionItem>

        <AccordionItem>
          <h2>
            <AccordionButton>Text Here</AccordionButton>
          </h2>
          <AccordionPanel>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
            eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim
            ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut
            aliquip ex ea commodo consequat.
          </AccordionPanel>
        </AccordionItem>

        <AccordionItem disabled>
          <h2>
            <AccordionButton>Text Here</AccordionButton>
          </h2>
          <AccordionPanel>Disabled panel</AccordionPanel>
        </AccordionItem>

        <AccordionItem>
          <h2>
            <AccordionButton>Text Here</AccordionButton>
          </h2>
          <AccordionPanel>
            <p className="mb-4">
              Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
              eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut
              enim ad minim veniam, quis nostrud exercitation ullamco laboris
              nisi ut aliquip ex ea commodo consequat.
            </p>

            <div className="flex space-x-4">
              <button type="button">Button</button>
              <button type="button">Button</button>
            </div>
          </AccordionPanel>
        </AccordionItem>
      </Accordion>
      <Accordion className="max-w-xs" defaultIndex={0}>
        <AccordionItem>
          <h2>
            <AccordionButton>Open by default</AccordionButton>
          </h2>
          <AccordionPanel>
            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
            eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim
            ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut
            aliquip ex ea commodo consequat.
          </AccordionPanel>
        </AccordionItem>
      </Accordion>
      <Accordion className="max-w-xs" defaultIndex={0}>
        <AccordionItem disabled>
          <h2>
            <AccordionButton>Disabled initially open</AccordionButton>
          </h2>
          <AccordionPanel>
            You can not open this accordion after closing it.
          </AccordionPanel>
        </AccordionItem>
        <AccordionItem>
          <h2>
            <AccordionButton>Click here</AccordionButton>
          </h2>
          <AccordionPanel>
            The disabled accordion is not focusable too.
          </AccordionPanel>
        </AccordionItem>
      </Accordion>
      <Accordion className="max-w-xs">
        <AccordionItem>
          <h2>
            <AccordionButton>Image example</AccordionButton>
          </h2>
          <AccordionPanel>
            <Image
              src="https://images.unsplash.com/photo-1623851467520-562e3d9b9e78?auto=format&fit=crop&w=701&q=80"
              ratio="4:3"
            />
          </AccordionPanel>
        </AccordionItem>
      </Accordion>
    </div>
  )
}

export const ControlledExample: Story<AccordionProps> = (args) => {
  const [active, setActive] = useState<number | null>(2)
  return (
    <Accordion
      className="max-w-2xl"
      index={active}
      onChange={setActive}
      {...args}
    >
      <AccordionItem>
        <h2>
          <AccordionButton>Input example</AccordionButton>
        </h2>

        <AccordionPanel>
          <p className="mb-4">
            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
            eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim
            ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut
            aliquip ex ea commodo consequat.
          </p>
          <div className="flex flex-col space-y-4">
            <input type="text" />
            <input type="text" />
            <input type="text" />
            <input type="text" />
          </div>
        </AccordionPanel>
      </AccordionItem>

      <AccordionItem>
        <AccordionButton>Lorem ipsum</AccordionButton>
        <AccordionPanel>
          Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
          eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad
          minim veniam, quis nostrud exercitation ullamco laboris nisi ut
          aliquip ex ea commodo consequat.
        </AccordionPanel>
      </AccordionItem>

      <AccordionItem disabled>
        <AccordionButton>I am not clickable or focusable</AccordionButton>
        <AccordionPanel>Disabled panel can be initially open. </AccordionPanel>
      </AccordionItem>

      <AccordionItem>
        <AccordionButton>Button here</AccordionButton>
        <AccordionPanel>
          <p className="mb-4">
            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
            eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim
            ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut
            aliquip ex ea commodo consequat.
          </p>

          <div className="flex space-x-4">
            <button type="button">Button</button>
            <button type="button">Button</button>
          </div>
        </AccordionPanel>
      </AccordionItem>
    </Accordion>
  )
}

export const AllCases: Story<AccordionProps> = (args) => {
  return (
    <div className="space-y-12">
      <div className="bg-white mt-8 p-4 rounded-md border border-dashed border-purple-400">
        <Example {...args} />
      </div>
      <div className="dark bg-gray-900 p-4 rounded-md border border-dashed border-purple-400">
        <Example {...args} />
      </div>
    </div>
  )
}

export default {
  title: 'Components/Accordion',
  component: Accordion,
  parameters: {
    controls: {
      include: ['size'],
    },
  },
} as Meta
