import React from 'react'
import cn from 'classnames'
import type {
  PropsWithRef,
  ComponentPropsWithoutRef,
  ComponentProps,
  LegacyRef,
} from 'react'
import { Checkmark, Indeterminate } from './Icons'

export interface Props extends ComponentPropsWithoutRef<'input'> {
  /**
   * The Checkbox label.
   *
   * All form controls should have labels.
   * When a label can't be used, it's necessary to add an
   * attribute directly to the input component.
   * In this case, you can apply one of the additional attributes.
   * (e.g. aria-label, aria-labelledby, title)
   * @see https://www.w3.org/TR/wai-aria-practices/#checkbox
   */
  label?: string
  /**
   * Defines the Checkbox as checked.
   * Indeterminate check has preference over this prop.
   */
  checked?: boolean
  /**
   * Defines the Checkbox as indeterminate.
   */
  indeterminate?: boolean
  /**
   * Defines the Checkbox as disabled.
   */
  disabled?: boolean
  labelProps?: ComponentProps<'label'>
  iconProps?: ComponentProps<'div'>
  /**
   * The input ref
   */
  inputRef?: LegacyRef<HTMLInputElement>
}

export type CheckboxProps = PropsWithRef<Props>

const defaultInputClasses =
  'opacity-0 absolute h-4 w-4 appearance-none border border-transparent cursor-pointer'
const defaultIconClasses = {
  main: 'border-2 rounded-sm w-4 h-4 flex justify-center items-center dark:text-gray-900',
  disabled:
    'bg-primary-100 text-primary-300 border-primary-100 dark:bg-primary-800 dark:border-primary-800',
  checked:
    'bg-primary-500 border-primary-500 text-white dark:bg-primary-400 dark:border-primary-400',
  unchecked: 'border-gray-400 dark:border-gray-200 bg-transparent',
}

function Checkbox({
  label,
  onChange,
  checked,
  indeterminate,
  disabled,
  inputRef,
  labelProps,
  iconProps,
  ...props
}: CheckboxProps) {
  const isChecked = checked || indeterminate
  const labelContainerClasses = cn(
    'text-gray-600 focus-within:text-gray-800 flex items-center h-6 cursor-pointer dark:text-gray-200 dark:focus-within:text-gray-100 font-roboto text-md',
    disabled && 'text-opacity-40 pointer-events-none'
  )
  const iconClasses = cn(defaultIconClasses.main, {
    [defaultIconClasses.disabled]: disabled,
    [isChecked ? defaultIconClasses.checked : defaultIconClasses.unchecked]:
      !disabled,
  })

  const Icon = indeterminate ? Indeterminate : Checkmark

  return (
    <label className={labelContainerClasses} {...labelProps}>
      <input
        ref={inputRef}
        type="checkbox"
        className={defaultInputClasses}
        checked={isChecked}
        onChange={onChange}
        disabled={disabled}
        aria-checked={indeterminate ? 'mixed' : Boolean(checked)}
        {...props}
      />
      <div className={iconClasses} {...iconProps}>
        <Icon className={cn(isChecked ? 'block' : 'hidden')} />
      </div>
      {label && <span className="ml-2">{label}</span>}
    </label>
  )
}

export { Checkbox }
