import React from 'react'
import cn from 'classnames'
import type {
  PropsWithRef,
  ComponentPropsWithoutRef,
  ComponentProps,
  LegacyRef,
  KeyboardEvent,
  MouseEvent,
} from 'react'
import { Keys } from '../../utils/keyboard'

export type SwitchSizeProp = 'sm' | 'md' | 'lg'

export interface Props extends Omit<ComponentPropsWithoutRef<'span'>, 'onChange'> {
  /**
   * The Switch size.
   * @defaultValue md
   */
  size?: SwitchSizeProp
  /**
   * Defines the Switch as checked.
   */
  checked?: boolean
  /**
   * Defines the Switch as disabled.
   */
  disabled?: boolean
  containerProps?: ComponentProps<'div'>
  /**
   * The input ref.
   */
  inputRef?: LegacyRef<HTMLInputElement>
  onChange?: (value: boolean) => void
}

export type SwitchProps = PropsWithRef<Props>

type Sizes = {
  [k in SwitchSizeProp]: string | string[]
}

const sizes: Sizes = {
  sm: 'w-7 h-4',
  md: 'w-9 h-5',
  lg: 'w-13 h-7',
}

const ballSizes: Sizes = {
  sm: 'w-3 h-3',
  md: 'w-4 h-4',
  lg: 'w-6 h-6',
}

function Switch({
  onChange,
  size = 'md',
  checked,
  disabled,
  inputRef,
  containerProps,
  ...props
}: SwitchProps) {
  const containerClasses = cn(
    'relative inline-flex items-center flex-shrink-0 transition-colors ease-in-out duration-200 cursor-pointer rounded-full border-2 border-transparent focus:outline-none',
    sizes[size],
    disabled && 'opacity-40 pointer-events-none',
    checked ? 'bg-primary-500 focus:bg-primary-400' : 'bg-gray-200 focus:bg-gray-100'
  )
  const ballClasses = cn(
    'absolute inline-block w- transform transition ease-in-out duration-200 rounded-full bg-white',
    ballSizes[size],
    checked ? 'translate-x-full' : 'translate-x-0'
  )

  function handleKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if ((e.key === Keys.Space || e.key === Keys.Enter) && !disabled) {
      e.stopPropagation()
      e.preventDefault()
      onChange?.(!checked)
    }
  }

  function handleClick(e: MouseEvent<HTMLDivElement>) {
    if (disabled) return
    e.preventDefault()
    e.stopPropagation()
    onChange?.(!checked)
  }

  return (
    <span
      className={containerClasses}
      onKeyDown={handleKeyDown}
      onClick={handleClick}
      tabIndex={disabled ? undefined : 0}
      aria-checked={!!checked}
      role="switch"
      {...props}
    >
      <input ref={inputRef} type="hidden" value={String(!!checked)} />
      <span className={ballClasses} aria-hidden="true" />
    </span>
  )
}

export { Switch }
