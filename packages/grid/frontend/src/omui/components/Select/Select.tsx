import React, {
  createContext,
  forwardRef,
  useCallback,
  useContext,
  useLayoutEffect,
  useRef,
  useState,
} from 'react'
import cn from 'classnames'
import { Keys } from '../../utils/keyboard'
import { useMergeRefs } from '../../hooks/useMergeRefs'
import { useOutsideClick } from '../../hooks/useOutsideClick'
import { Chevron, Checkmark } from './Icons'
import { Text } from '@/omui'
import type {
  PropsWithRef,
  KeyboardEvent as ReactKeyboardEvent,
  MouseEvent as ReactMouseEvent,
} from 'react'

export type SelectSizeProp = 'sm' | 'md' | 'lg'
export type SelectOption = { label: string; value: OptionValue; disabled?: boolean }
export type OptionValue = string | number | null

export interface Props {
  /**
   * The size of the Select.
   * @defaultValue md
   */
  size?: SelectSizeProp
  /**
   * The placeholder of the Select.
   * @defaultValue Select option
   */
  placeholder?: string
  /**
   * The selected value of the Select.
   */
  value?: string | number | null
  /**
   * The change handler function to the Select.
   */
  onChange?: (option: OptionValue) => void
  /**
   * The options of the Select.
   */
  options?: SelectOption[]
  /**
   * Set the Select with error state.
   */
  error?: boolean
  /**
   * Defines the Select as disabled.
   */
  disabled?: boolean
  label?: string
  required?: boolean
}

export type SelectProps = PropsWithRef<Props>

type SelectContextValues = {
  value?: OptionValue
  closeMenu?: () => void
  onKeyboardNavigate?: (id: string) => void
}
const SelectContext = createContext<SelectContextValues>({})

const defaultClasses =
  'font-roboto px-3.5 bg-white relative border rounded focus:shadow-primary-focus focus:outline-none dark:bg-gray-700 dark:text-white'
const optionsMenuClasses =
  'absolute font-roboto top-0 -mt-1 left-0 bg-primary-50 shadow w-full py-2 overflow-y-scroll max-h-60 dark:bg-gray-800'

type Sizes = {
  [k in SelectSizeProp]: string
}

const sizes: Sizes = {
  sm: 'py-1 text-sm',
  md: 'py-2 text-md',
  lg: 'py-3 text-lg',
}

type EventCases = {
  [k in Keys]?: () => void
}

const Select = forwardRef<HTMLDivElement, SelectProps>(function Select(
  {
    size = 'md',
    placeholder = 'Select option',
    label,
    required,
    disabled,
    value,
    onChange,
    options,
    error,
    ...props
  },
  ref
) {
  const innerRef = useRef<HTMLDivElement>(null)
  const [isOpen, setIsOpen] = useState<boolean>(false)
  const [focusedElement, setFocusedElement] = useState<string | null>(null)

  const selectClasses = cn(
    defaultClasses,
    {
      'opacity-40 pointer-events-none': disabled,
      'border-error-500 dark:border-error-400': error,
      'border-gray-100 dark:border-gray-800': !error,
    },
    sizes[size]
  )

  const handleClick = (e: ReactMouseEvent<HTMLDivElement>) => {
    if (disabled) return
    e.preventDefault()
    e.stopPropagation()
    if (!isOpen) {
      setIsOpen(true)
    } else {
      handleCloseMenu()
    }
  }

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (disabled) return
    const eventCases: EventCases = {
      [Keys.Space]: () => setIsOpen(true),
      [Keys.Enter]: () => setIsOpen(true),
      [Keys.Escape]: () => handleCloseMenu(),
    }

    if (eventCases[event.key]) {
      event.preventDefault()
      event.stopPropagation()
      eventCases[event.key]()
    }
  }

  function handleCloseMenu() {
    setIsOpen(false)
    setFocusedElement(null)
    innerRef.current?.focus()
  }

  function handleClickOption(option: OptionValue) {
    if (onChange) onChange(option)
    handleCloseMenu()
  }

  useOutsideClick({
    ref: innerRef,
    callback: () => {
      if (isOpen) {
        handleCloseMenu()
      }
    },
  })

  /**
   * When the menu opens it focus the selected value
   * or the placeholder when value is not set
   */
  useLayoutEffect(() => {
    if (isOpen) {
      document.getElementById(`omui-menu-item-${value ?? ''}`)?.focus()
    }
  }, [value, isOpen])

  const activeLabel =
    value !== undefined || value !== null ? options?.find(i => i.value === value)?.label : null

  const handleKeyboardNavigate = useCallback((focusedId: string) => {
    if (focusedId) {
      document.getElementById(focusedId)?.focus()
      setFocusedElement(focusedId)
    }
  }, [])

  return (
    <SelectContext.Provider
      value={{ value, closeMenu: handleCloseMenu, onKeyboardNavigate: handleKeyboardNavigate }}
    >
      <div className="space-y-2">
        {label && (
          <label htmlFor={props.name || props.id}>
            <Text>{label}</Text> {/* TODO: sub below with required Icon */}
            {required && (
              <Text size="sm" className="text-primary-500">
                *
              </Text>
            )}
          </label>
        )}
        <div
          className={selectClasses}
          onClick={handleClick}
          onKeyDown={handleKeyDown}
          ref={useMergeRefs(ref, innerRef)}
          tabIndex={disabled ? undefined : 0}
          aria-haspopup={true}
          aria-expanded={isOpen ? true : undefined}
          data-active={isOpen ? true : undefined}
          aria-owns={isOpen ? 'omui-menu-list' : undefined}
          aria-activedescendant={focusedElement ?? undefined}
          id={props.name || props.id}
          {...props}
        >
          <span className="flex space-x-4 justify-between items-center">
            <span>{activeLabel || placeholder}</span>
            <Chevron className="text-gray-500 dark:text-gray-200" />
          </span>

          {isOpen && options?.length ? (
            <ul id="omui-menu-list" role="menu" className={optionsMenuClasses}>
              <Option
                option={{ label: placeholder, value: '', placeholder: true, disabled: true }}
              />
              {options.map(option => (
                <Option key={option.value as string} option={option} onClick={handleClickOption} />
              ))}
            </ul>
          ) : null}
        </div>
      </div>
    </SelectContext.Provider>
  )
})

type OptionProps = {
  option: SelectOption & { placeholder?: boolean }
  onClick?: (option: OptionValue) => void
}

function Option({ option, onClick }: OptionProps) {
  const { value, closeMenu, onKeyboardNavigate } = useContext(SelectContext)
  const optionId = `omui-menu-item-${option.value}`
  const selectedValue = value === option.value
  const noActiveValue = (value === undefined || value === null) && option.placeholder
  const active = selectedValue || noActiveValue

  const optionClasses = cn(
    'flex items-center relative py-0.5 my-0.5 px-6 cursor-pointer focus:outline-none',
    'focus:bg-primary-500 hover:text-white hover:bg-primary-500 focus:text-white dark:focus:bg-primary-600 dark:hover:bg-primary-600 text-gray-800 dark:text-white',
    { 'text-opacity-50': option.disabled }
  )

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLLIElement>, option: SelectOption) => {
    const eventCases: EventCases = {
      [Keys.ArrowDown]: () => {
        const nextSibling = document.getElementById(optionId)?.nextElementSibling?.id
        if (nextSibling) return onKeyboardNavigate?.(nextSibling)
      },
      [Keys.ArrowUp]: () => {
        const prevSibling = document.getElementById(optionId)?.previousElementSibling?.id
        if (prevSibling) return onKeyboardNavigate?.(prevSibling)
      },
      [Keys.Space]: () => onClick?.(option.value),
      [Keys.Enter]: () => onClick?.(option.value),
      [Keys.Escape]: () => closeMenu?.(),
      [Keys.Tab]: () => null,
    }
    if (eventCases[event.key]) {
      event.preventDefault()
      event.stopPropagation()
      eventCases[event.key]()
    }
  }

  function handleClickOption(event: ReactMouseEvent<HTMLLIElement>, option: SelectOption) {
    if (option.disabled) return event.preventDefault()
    if (onClick) return onClick(option.value)
  }

  return (
    <li
      id={optionId}
      className={optionClasses}
      onClick={e => handleClickOption(e, option)}
      role="menuitem"
      tabIndex={-1}
      onKeyDown={e => handleKeyDown(e, option)}
      aria-disabled={option.disabled === true ? true : undefined}
    >
      {active && (
        <span className="absolute left-1.5">
          <Checkmark />
        </span>
      )}
      {option.label}
    </li>
  )
}

export { Select }
