import { Children } from 'react'
import cn from 'classnames'
import { Disclosure } from '@headlessui/react'
import { ChevronRightIcon } from '@heroicons/react/outline'
import AnimateHeight from 'react-animate-height'
import type { ReactNode } from 'react'

function AccordionSummary({ isOpen, openClasses, children }) {
  return (
    <div
      className={cn(
        'cursor-pointer px-4 py-3 flex items-center space-x-2',
        isOpen && openClasses
      )}
    >
      <ChevronRightIcon
        className={cn(
          'mx-2 my-1 w-6 h-6 text-gray-500 transition transform',
          isOpen && 'rotate-90'
        )}
      />
      <div className="flex items-center justify-between w-full">{children}</div>
    </div>
  )
}

// todo: untangle classes
export function AccordionListItem({
  openClasses = '',
  children,
}: {
  openClasses?: string
  children: ReactNode
}) {
  const [summary, ...panel] = Children.toArray(children)

  return (
    <div role="region">
      <Disclosure>
        {({ open }) => (
          <>
            <Disclosure.Button as="div" className="w-full">
              <AccordionSummary openClasses={openClasses} isOpen={open}>
                {summary}
              </AccordionSummary>
            </Disclosure.Button>
            <AnimateHeight duration={200} height={open ? 'auto' : 0}>
              <Disclosure.Panel static>
                <div className={cn('p-4 pb-8', open && openClasses)}>
                  {panel}
                </div>
              </Disclosure.Panel>
            </AnimateHeight>
          </>
        )}
      </Disclosure>
    </div>
  )
}

function AccordionRoot({ children }) {
  return (
    <div className="grid grid-cols-1 divide-y divide-gray-200 border border-gray-200 rounded-md">
      {children}
    </div>
  )
}

export const Accordion = Object.assign(AccordionRoot, {
  Item: AccordionListItem,
})
