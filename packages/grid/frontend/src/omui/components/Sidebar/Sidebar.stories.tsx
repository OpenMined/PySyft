import React from 'react'
import { Story, Meta } from '@storybook/react'
import { Sidebar } from './Sidebar'
import type { SidebarProps } from './Sidebar'

import {
  List,
  ListAvatarItem,
  ListContainedItem,
  ListIconItem,
  ListItem,
  UnorderedList,
  UnorderedListItem,
} from '../List/List'
import { Divider } from '../Divider/Divider'
import { Text } from '../Typography/Text'

export default {
  title: 'Components/Sidebar',
  component: Sidebar,
  parameters: {
    controls: {
      include: ['children', 'header'],
    },
  },
  argTypes: {
    children: { name: 'children', defaultValue: 'Text Here', control: { type: 'text' } },
  },
} as Meta

export const Default: Story<SidebarProps> = args => (
  <Sidebar {...args}>
    <List>
      {Array.from(Array(10).keys()).map((_, index) => (
        <ListItem key={`def-${index}`}>{args.children}</ListItem>
      ))}
    </List>
  </Sidebar>
)

const ItemCase = ({ disabled, children }: any) => (
  <Sidebar>
    <List>
      {Array.from(Array(6).keys()).map((_, index) => (
        <button disabled={index === 1 && disabled} className="block w-full">
          <ListItem key={`def-${index}`}>{children}</ListItem>
        </button>
      ))}
    </List>
  </Sidebar>
)
const AvatarCase = ({ disabled, children }: any) => (
  <Sidebar>
    <List>
      {Array.from(Array(6).keys()).map((_, index) => (
        <button disabled={index === 1 && disabled} className="block w-full">
          <ListAvatarItem
            key={`a-${index}`}
            src="https://images.unsplash.com/photo-1623288749528-e40a033da0f7"
          >
            {children}
          </ListAvatarItem>
        </button>
      ))}
    </List>
  </Sidebar>
)
const IconCase = ({ disabled, children }: any) => (
  <Sidebar>
    <List>
      {Array.from(Array(6).keys()).map((_, index) => (
        <button disabled={index === 1 && disabled} className="block w-full">
          <ListIconItem icon={RandomIcon} key={`ic-${index}`}>
            {children}
          </ListIconItem>
        </button>
      ))}
    </List>
  </Sidebar>
)
const ContainedCase = ({ disabled, children }: any) => (
  <Sidebar>
    <List>
      {Array.from(Array(6).keys()).map((_, index) => (
        <button disabled={index === 1 && disabled} className="block w-full">
          <ListContainedItem containedValue={index + 1} key={`con-${index}`}>
            {children}
          </ListContainedItem>
        </button>
      ))}
    </List>
    <Divider color="light" className="my-4 dark:border-gray-700" />
    <div className="text-gray-600 dark:text-white">
      <svg
        width="29"
        height="24"
        viewBox="0 0 29 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M20.5 9C20.5 4.875 16.0938 1.5 10.75 1.5C5.35938 1.5 1 4.875 1 9C1 10.6406 1.65625 12.0938 2.78125 13.3125C2.125 14.7656 1.09375 15.8906 1.09375 15.8906C1 15.9844 0.953125 16.1719 1 16.3125C1.09375 16.4531 1.1875 16.5 1.375 16.5C3.0625 16.5 4.46875 15.9375 5.5 15.3281C7 16.0781 8.82812 16.5 10.75 16.5C16.0938 16.5 20.5 13.1719 20.5 9ZM26.2188 19.3125C27.2969 18.0938 28 16.6406 28 15C28 11.9062 25.4688 9.1875 21.9062 8.0625C21.9531 8.39062 22 8.71875 22 9C22 13.9688 16.9375 18 10.75 18C10.2344 18 9.71875 18 9.25 17.9531C10.7031 20.625 14.1719 22.5 18.25 22.5C20.1719 22.5 21.9531 22.0781 23.4531 21.3281C24.4844 21.9375 25.8906 22.5 27.625 22.5C27.7656 22.5 27.9062 22.4531 27.9531 22.3125C28 22.1719 28 21.9844 27.8594 21.8906C27.8594 21.8906 26.8281 20.7656 26.2188 19.3125Z"
          fill="currentColor"
        />
      </svg>
      <Text className="mt-5">
        Not seeing an answer to your specific question? Go to our discussion board to get extra
        assistance.
      </Text>
    </div>
  </Sidebar>
)
const FullCase = ({ disabled, children }: any) => (
  <Sidebar>
    <List>
      {Array.from(Array(6).keys()).map((_, index) => (
        <button disabled={index === 1 && disabled} className="block w-full">
          <ListIconItem icon={RandomIcon} key={`ic-${index}`}>
            {children}
          </ListIconItem>
        </button>
      ))}
    </List>
    <Divider color="light" className="my-8 dark:border-gray-700" />
    <div className="text-gray-600 dark:text-white">
      <svg
        width="29"
        height="24"
        viewBox="0 0 29 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M20.5 9C20.5 4.875 16.0938 1.5 10.75 1.5C5.35938 1.5 1 4.875 1 9C1 10.6406 1.65625 12.0938 2.78125 13.3125C2.125 14.7656 1.09375 15.8906 1.09375 15.8906C1 15.9844 0.953125 16.1719 1 16.3125C1.09375 16.4531 1.1875 16.5 1.375 16.5C3.0625 16.5 4.46875 15.9375 5.5 15.3281C7 16.0781 8.82812 16.5 10.75 16.5C16.0938 16.5 20.5 13.1719 20.5 9ZM26.2188 19.3125C27.2969 18.0938 28 16.6406 28 15C28 11.9062 25.4688 9.1875 21.9062 8.0625C21.9531 8.39062 22 8.71875 22 9C22 13.9688 16.9375 18 10.75 18C10.2344 18 9.71875 18 9.25 17.9531C10.7031 20.625 14.1719 22.5 18.25 22.5C20.1719 22.5 21.9531 22.0781 23.4531 21.3281C24.4844 21.9375 25.8906 22.5 27.625 22.5C27.7656 22.5 27.9062 22.4531 27.9531 22.3125C28 22.1719 28 21.9844 27.8594 21.8906C27.8594 21.8906 26.8281 20.7656 26.2188 19.3125Z"
          fill="currentColor"
        />
      </svg>
      <Text className="mt-5">
        Not seeing an answer to your specific question? Go to our discussion board to get extra
        assistance.
      </Text>
    </div>
    <Divider color="light" className="my-8 dark:border-gray-700" />
    <Text bold className="mb-4 dark:text-white text-gray-800">
      Pre-Requisites
    </Text>
    <UnorderedList>
      {Array.from(Array(6).keys()).map((_, index) => (
        <a className="block w-full">
          <UnorderedListItem key={`uli-${index}`}>{children}</UnorderedListItem>
        </a>
      ))}
    </UnorderedList>
    <Divider color="light" className="my-8 dark:border-gray-700" />
    <List>
      {Array.from(Array(6).keys()).map((_, index) => (
        <button disabled={index === 1 && disabled} className="block w-full">
          <ListIconItem icon={RandomIcon} key={`ic-${index}`}>
            {children}
          </ListIconItem>
        </button>
      ))}
    </List>
  </Sidebar>
)

const Example = (props: any) => {
  return (
    <div className="flex flex-col space-y-4">
      <div className="flex space-x-6">
        <ItemCase {...props} />
        <ItemCase disabled {...props} />
        <AvatarCase {...props} />
        <AvatarCase disabled {...props} />
        <IconCase {...props} />
        <IconCase disabled {...props} />
      </div>
      <div className="flex space-x-6">
        <ContainedCase {...props} />
        <ContainedCase disabled {...props} />
        <FullCase {...props} />
        <FullCase disabled {...props} />
      </div>
    </div>
  )
}

export const AllCases: Story<SidebarProps> = args => (
  <div className="flex flex-col space-y-10">
    <div className="flex space-x-8 bg-white p-4 rounded-md border border-dashed border-purple-400">
      <Example>{args.children}</Example>
    </div>
    <div className="dark flex space-x-8 bg-gray-900 p-4 rounded-md border border-dashed border-purple-400">
      <Example>{args.children}</Example>
    </div>
  </div>
)

const RandomIcon = ({ className }: { className: string }) => (
  <svg className={className} role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 496 512">
    <path
      fill="currentColor"
      d="M248 8C111 8 0 119 0 256s111 248 248 248 248-111 248-248S385 8 248 8zm80 168c17.7 0 32 14.3 32 32s-14.3 32-32 32-32-14.3-32-32 14.3-32 32-32zm-160 0c17.7 0 32 14.3 32 32s-14.3 32-32 32-32-14.3-32-32 14.3-32 32-32zm194.8 170.2C334.3 380.4 292.5 400 248 400s-86.3-19.6-114.8-53.8c-13.6-16.3 11-36.7 24.6-20.5 22.4 26.9 55.2 42.2 90.2 42.2s67.8-15.4 90.2-42.2c13.4-16.2 38.1 4.2 24.6 20.5z"
    />
  </svg>
)
