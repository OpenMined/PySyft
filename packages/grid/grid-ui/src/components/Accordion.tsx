import {Children} from 'react'
import cn from 'classnames'
import {Disclosure} from '@headlessui/react'
import {ChevronRightIcon} from '@heroicons/react/outline'
import AnimateHeight from 'react-animate-height'
import type {ReactNode} from 'react'

function AccordionSummary({isOpen, children}) {
  return (
    <div className="cursor-pointer px-4 py-5 flex items-center space-x-2 bg-white hover:bg-sky-100">
      <ChevronRightIcon className={cn('mx-2 my-1 w-6 h-6 text-gray-500 transition transform', isOpen && 'rotate-90')} />
      <div className="flex items-center justify-between w-full">{children}</div>
    </div>
  )
}

export function AccordionListItem({children}: {children: ReactNode}) {
  const [summary, ...panel] = Children.toArray(children)
  return (
    <div role="region">
      <Disclosure>
        {({open}) => (
          <>
            <h3>
              <Disclosure.Button as="div" className="w-full">
                <AccordionSummary isOpen={open}>{summary}</AccordionSummary>
              </Disclosure.Button>
            </h3>
            <AnimateHeight duration={200} height={open ? 'auto' : 0}>
              <Disclosure.Panel static>{panel}</Disclosure.Panel>
            </AnimateHeight>
          </>
        )}
      </Disclosure>
    </div>
  )
}

function AccordionRoot({children}) {
  return <div className="grid grid-cols-1 divide-y divide-gray-200 border border-gray-200">{children}</div>
}

export const Accordion = Object.assign(AccordionRoot, {Item: AccordionListItem})
