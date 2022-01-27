import { Fragment } from 'react'
import tw from 'twin.macro'
import { Popover as HeadlessPopover } from '@headlessui/react'
import Transition from './Transition'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import type { PropsWithChildren } from 'react'

type PopoverProps = {
  label?: string
}

type TriggerProps = {
  label?: string
  open: boolean
}

export const Popover = ({ label, children }: PropsWithChildren<PopoverProps>) => {
  return (
    <HeadlessPopover tw="relative">
      {({ open }) => (
        <>
          <Trigger label={label} open={open} />
          <Transition as={Fragment} {...transitionProps}>
            <HeadlessPopover.Panel tw="absolute z-10 w-screen max-w-sm mt-3 transform left-0 sm:px-0">
              {children}
            </HeadlessPopover.Panel>
          </Transition>
        </>
      )}
    </HeadlessPopover>
  )
}

const Trigger = ({ label, open }: TriggerProps) => {
  return (
    <HeadlessPopover.Button
      className="group"
      css={[
        tw`inline-flex items-center focus:outline-none`,
        tw`text-base font-medium text-current p-2`,
        !open && tw`text-opacity-90`,
      ]}
    >
      <span>{label}</span>
      <FontAwesomeIcon icon="chevron-down" />
    </HeadlessPopover.Button>
  )
}

const transitionProps = {
  enter: tw`transition ease-out duration-200`,
  enterFrom: tw`opacity-0 translate-y-1`,
  enterTo: tw`opacity-100 translate-y-0`,
  leave: tw`transition ease-in duration-150`,
  leaveFrom: tw`opacity-100 translate-y-0`,
  leaveTo: tw`opacity-0 translate-y-1`,
}
