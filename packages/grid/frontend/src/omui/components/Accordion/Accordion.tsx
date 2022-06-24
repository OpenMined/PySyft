import React, { useState, useContext, createContext, forwardRef } from 'react'
import type {
  PropsWithRef,
  ButtonHTMLAttributes,
  HTMLProps,
  ReactNode,
} from 'react'
import cn from 'classnames'
import AnimateHeight from 'react-animate-height'

import { Chevron } from './Icons/Chevron'
import { Text, TextSizeProp } from '../Typography/Text'

export type AccordionSizeProp = 'sm' | 'md' | 'lg'

export interface Props {
  /**
   * The size of the Accordion.
   * @defaultValue sm
   */
  size?: AccordionSizeProp
  /**
   * The index(es) of the expanded Accordion item;
   */
  index?: number | null
  /**
   * The initial index(es) of the expanded Accordion item;
   */
  defaultIndex?: number
  /**
   * The callback invoked when Accordion items are expanded or collapsed.
   */
  onChange?(expandedIndex: number | null): void
  className?: string
  children: ReactNode
}

export type AccordionProps = PropsWithRef<Props>

type Sizes<T> = {
  [k in AccordionSizeProp]: T
}

const defaultSize = 'sm'

type AccordionContextValues = {
  activeIndex: number | null
  onChange?: AccordionProps['onChange']
  size: AccordionSizeProp
}
const AccordionContext = createContext<AccordionContextValues>({
  activeIndex: -1,
  size: defaultSize,
})

type AccordionItemContextValues = {
  id: number
  isOpen: boolean
  isDisabled: boolean
}
const AccordionItemContext = createContext<AccordionItemContextValues>({
  id: -1,
  isOpen: false,
  isDisabled: false,
})

const Accordion = forwardRef<HTMLDivElement, AccordionProps>(function Accordion(
  {
    size = defaultSize,
    index,
    defaultIndex = -1,
    onChange: onChangeProp,
    className,
    children,
    ...props
  },
  ref
) {
  const isControlled = index !== undefined
  const [valueState, setValueState] = useState<number | null>(
    index ?? defaultIndex
  )
  const value = isControlled ? index ?? -1 : valueState
  function onChange(clickedValue: number) {
    const next = clickedValue === value ? null : clickedValue
    if (!isControlled) {
      setValueState(next)
    }
    onChangeProp?.(next)
  }

  return (
    <div ref={ref} className={cn('w-full', className)} {...props}>
      <AccordionContext.Provider value={{ activeIndex: value, onChange, size }}>
        {React.Children.map(children, (child, index) => {
          if (React.isValidElement(child)) {
            return React.cloneElement(child, { index })
          }
          return child
        })}
      </AccordionContext.Provider>
    </div>
  )
})

type AccordionItemProps = {
  children: ReactNode
  disabled?: boolean
  index?: number
}

const AccordionItem = ({
  children,
  disabled,
  index = -1,
}: AccordionItemProps) => {
  const context = useContext(AccordionContext)
  const isOpen = context.activeIndex === index
  const itemClasses = cn(
    'border-gray-200 border-t border-b',
    disabled && 'opacity-40 pointer-events-none'
  )

  return (
    <AccordionItemContext.Provider
      value={{ id: index, isOpen, isDisabled: !!disabled }}
    >
      <div className={itemClasses} id={`omui-accordion-item-${index}`}>
        {children}
      </div>
    </AccordionItemContext.Provider>
  )
}

const textSizes: Sizes<TextSizeProp> = {
  sm: 'lg',
  md: 'xl',
  lg: '2xl',
}

/**
 * AccordionButton is the button used to expand and collapse an accordion item.
 * It must be a child of `AccordionItem`.
 *
 * Note: Each accordion button must be wrapped in an heading tag,
 * that is appropriate for the information architecture of the page
 * and improved accessibility.
 */
const AccordionButton = ({
  children,
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement>) => {
  const { size, onChange } = useContext(AccordionContext)
  const { id, isDisabled, isOpen } = useContext(AccordionItemContext)

  const buttonClasses = cn(
    'flex justify-between items-center w-full focus:shadow-primary-focus p-2',
    className
  )

  return (
    <button
      aria-expanded={isOpen}
      aria-controls={`omui-accordion-panel-${id}`}
      className={buttonClasses}
      id={`omui-accordion-button-${id}`}
      disabled={isDisabled}
      onClick={() => onChange?.(id)}
      {...props}
    >
      <Text
        as="span"
        size={textSizes[size]}
        className="text-left text-gray-800 dark:text-white flex items-center"
      >
        {children}
      </Text>
      <AccordionIcon size={size} isOpen={isOpen} />
    </button>
  )
}

/**
 * AccordionPanel is the container that display the Accordion content.
 * It must be a child of `AccordionItem`.
 */
const AccordionPanel = ({ className, ...props }: HTMLProps<HTMLDivElement>) => {
  const { id, isOpen } = useContext(AccordionItemContext)
  const panelClasses = cn('p-4 text-gray-800 dark:text-gray-200', className)

  return (
    <AnimateHeight
      id={`omui-accordion-panel-${id}`}
      height={isOpen ? 'auto' : 0}
    >
      <div
        className={panelClasses}
        role="region"
        aria-labelledby={`omui-accordion-button-${id}`}
        {...props}
      />
    </AnimateHeight>
  )
}

type AccordionIconProps = {
  size: AccordionSizeProp
  isOpen: boolean
}

function AccordionIcon({ size, isOpen }: AccordionIconProps) {
  const iconClasses = cn(
    'transform transition-transform origin-center text-gray-500 dark:text-gray-400',
    !isOpen && '-rotate-180',
    { sm: 'w-2.5', md: 'w-3', lg: 'w-4' }[size]
  )

  return (
    <Chevron className={iconClasses} aria-hidden={true} focusable={false} />
  )
}

export { Accordion, AccordionItem, AccordionButton, AccordionPanel }
