import React, { createContext, forwardRef, useContext, useEffect, useMemo } from 'react'
import cn from 'classnames'
import type {
  ReactNode,
  PropsWithRef,
  KeyboardEvent as ReactKeyboardEvent,
  MouseEvent as ReactMouseEvent,
} from 'react'

import { Keys } from '../../utils/keyboard'

export type TabsVariantProp = 'outline' | 'underline'
export type TabsSizeProp = 'sm' | 'md' | 'lg' | 'xl'
export type TabsAlignProp = 'left' | 'right' | 'auto'
export type TabsListProp = { id: number | string; title: ReactNode | string; disabled?: boolean }

export interface Props {
  /**
   * The variant of Tabs.
   * @defaultValue outline
   */
  variant?: TabsVariantProp
  /**
   * The size of Tabs.
   * @defaultValue md
   */
  size?: TabsSizeProp
  /**
   * The alignment of Tab list.
   * @defaultValue auto
   */
  align?: TabsAlignProp
  /**
   * The tabs list to be rendered.
   */
  tabsList: TabsListProp[]
  /**
   * The active Tab.
   * Must be a id from `tabsList`.
   */
  active?: TabsListProp['id']
  /**
   * The click handler function to Tabs.
   */
  onChange: (id: TabsListProp['id']) => void
  /**
   * The children to be rendered inside a div with role `tabpanel`
   */
  children?: ReactNode
}

export type TabsProps = PropsWithRef<Props>

type EventCases = {
  [k in Keys]?: () => void
}

type Alignments = {
  [k in TabsAlignProp]?: string
}

type Sizes = {
  [k in TabsSizeProp]: string
}

type Variants = {
  [k in TabsVariantProp]: {
    default: string
    active: string
    inactive: string
    container: string
  }
}

const alignments: Alignments = {
  left: 'justify-start',
  right: 'justify-end',
}

const sizes: Sizes = {
  sm: 'py-2 px-2.5 text-sm',
  md: 'py-2.5 px-4 text-md',
  lg: 'py-4 px-8 text-lg',
  xl: 'py-4.5 px-8 text-xl',
}

const variants: Variants = {
  outline: {
    default: 'border-primary-500',
    active: 'bg-white border-l-2 border-r-2 border-t-2 rounded-t',
    inactive: 'border-b-2 mt-1 mx-0.5',
    container: 'border-primary-500',
  },
  underline: {
    default: 'bg-transparent border-b-2',
    active: 'border-primary-500',
    inactive: 'border-gray-200',
    container: 'border-gray-200',
  },
}

const defaultSize = 'md'
const defaultVariant = 'outline'
const defaultAlign = 'auto'

type TabsContextValues = {
  onChange?: TabsProps['onChange']
  focusNext?: TabsProps['onChange']
  focusPrev?: TabsProps['onChange']
  size: TabsSizeProp
  variant: TabsVariantProp
  align: TabsAlignProp
}
const TabsContext = createContext<TabsContextValues>({
  size: defaultSize,
  variant: defaultVariant,
  align: defaultAlign,
})

const Tabs = forwardRef<HTMLDivElement, TabsProps>(function Tabs(
  {
    align = defaultAlign,
    size = defaultSize,
    variant = defaultVariant,
    tabsList,
    active,
    onChange,
    children,
    ...props
  },
  ref
) {
  const enabledTabs = useMemo(() => tabsList.filter(i => !i.disabled), [tabsList])

  /**
   * If no active tab, set the first and not disabled tab
   */
  useEffect(() => {
    if (!active) {
      onChange(enabledTabs[0].id)
    }
  }, [active, onChange, enabledTabs])

  function focusAndChange(elementId: TabsListProp['id']) {
    document.getElementById(`omui-tab-${elementId}`)?.focus()
    onChange(elementId)
  }

  function handleNextEl(id: TabsListProp['id']) {
    // find next element or first of the enabled tabs list
    const nextEl = (enabledTabs[enabledTabs.findIndex(i => i.id === id) + 1] || enabledTabs[0]).id
    focusAndChange(nextEl)
  }

  function handlePrevEl(id: TabsListProp['id']) {
    // find previous element or last of the enabled tabs list
    const prevEl = (
      enabledTabs[enabledTabs.findIndex(i => i.id === id) - 1] ||
      enabledTabs[enabledTabs.length - 1]
    ).id

    focusAndChange(prevEl)
  }

  const tabListClasses = cn(
    'font-roboto border-b-2 flex w-full',
    variants[variant]?.container,
    alignments[align]
  )

  return (
    <TabsContext.Provider
      value={{ focusNext: handleNextEl, focusPrev: handlePrevEl, variant, onChange, size, align }}
    >
      <div ref={ref} {...props}>
        <div className={tabListClasses} role="tablist" aria-orientation="horizontal">
          {tabsList?.map(tab => (
            <Tab key={tab.id} {...tab} isActive={active === tab.id} />
          ))}
        </div>
        {active !== null && (
          <div role="tabpanel" aria-labelledby={`omui-tab-${active}`}>
            {children}
          </div>
        )}
      </div>
    </TabsContext.Provider>
  )
})

export type TabProps = { isActive?: boolean } & TabsListProp

function Tab({ id, title, disabled, isActive }: TabProps) {
  const context = useContext(TabsContext)
  const tabId = `omui-tab-${id}`

  const tabClasses = cn(
    ['relative top-0.5 font-roboto font-bold', variants[context.variant]?.default],
    isActive
      ? ['text-primary-600', variants[context.variant]?.active]
      : ['text-gray-400', variants[context.variant]?.inactive],
    sizes[context.size],
    context.align === 'auto' ? 'w-full' : 'w-auto',
    { 'text-opacity-40 pointer-events-none': disabled }
  )

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLButtonElement>) => {
    const eventCases: EventCases = {
      [Keys.ArrowRight]: () => context.focusNext?.(id),
      [Keys.ArrowLeft]: () => context.focusPrev?.(id),
      [Keys.Enter]: () => context.onChange?.(id),
      [Keys.Space]: () => context.onChange?.(id),
    }

    if (eventCases[event.key]) {
      event.preventDefault()
      event.stopPropagation()
      eventCases[event.key]()
    }
  }

  function handleClickOption(event: ReactMouseEvent<HTMLButtonElement>) {
    if (disabled) return event.preventDefault()
    if (context.onChange) return context.onChange(id)
  }

  return (
    <button
      id={tabId}
      className={tabClasses}
      onClick={handleClickOption}
      onKeyDown={handleKeyDown}
      role="tab"
      tabIndex={isActive ? 0 : -1}
      disabled={disabled ? true : undefined}
      aria-selected={isActive}
    >
      {title}
    </button>
  )
}

export { Tabs }
