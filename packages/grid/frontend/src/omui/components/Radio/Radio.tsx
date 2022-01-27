import React from 'react'
import cn from 'classnames'
import type { PropsWithRef, ComponentPropsWithoutRef, ComponentProps, LegacyRef } from 'react'

import { useRadioGroupContext } from './RadioGroup'

export type RadioSizeProp = 'sm' | 'md' | 'lg'

export interface Props extends ComponentPropsWithoutRef<'input'> {
  /**
   * The name attribute of the Radio.
   * Inherits the RadioGroup name when there is one.
   */
  name?: string
  /**
   * The checked attribute of the Radio.
   */
  checked?: boolean
  /**
   * The disabled attribute of the Radio.
   */
  disabled?: boolean
  /**
   * The label of the Radio.
   *
   * All form controls should have labels.
   * When a label can't be used, it's necessary to add an
   * attribute directly to the input component.
   * In this case, you can apply the additional attribute
   * via the inputProps property.
   * (e.g. aria-label, aria-labelledby, title)
   */
  label?: string
  containerProps?: ComponentProps<'label'>
  /**
   * The value attribute of the Radio.
   */
  value?: string | number | readonly string[]
  /**
   * The input ref.
   */
  inputRef?: LegacyRef<HTMLInputElement>
}

export type RadioProps = PropsWithRef<Props>

function Radio({
  onChange: onChangeProp,
  name: nameProp,
  label,
  checked,
  value,
  disabled,
  inputRef,
  containerProps,
  ...props
}: RadioProps) {
  const group = useRadioGroupContext()
  const inputClasses = cn(
    'text-primary-400 border-2 focus:ring-offset-0 focus:ring-0 focus:outline-none',
    disabled && 'opacity-40 pointer-events-none',
    checked
      ? 'bg-primary-500 border-primary-500 dark:border-primary-400 dark:bg-primary-400 dark:bg-radio'
      : 'border-gray-400 bg-transparent'
  )
  const labelClasses = cn('text-gray-600 dark:text-gray-200 ml-2', disabled && 'opacity-40')

  /**
   * Checked value and onChange are inherited
   * from RadioGroup context when it exists.
   */
  let isChecked = checked
  if (group?.value && group.value !== null) {
    isChecked = group.value === value
  }

  let onChange = onChangeProp
  if (group?.onChange && value !== null) {
    onChange = e => {
      group.onChange(e.target.value)
      onChangeProp?.(e)
    }
  }

  const name = nameProp ?? group?.name

  return (
    <label className="flex items-center" {...containerProps}>
      <input
        name={name}
        ref={inputRef}
        type="radio"
        value={value}
        checked={isChecked}
        disabled={disabled}
        className={inputClasses}
        onChange={onChange}
        {...props}
      />
      {label && <span className={labelClasses}>{label}</span>}
    </label>
  )
}

export { Radio }
