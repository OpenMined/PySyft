import React, { forwardRef } from 'react'
import cn from 'classnames'
import type {
  ReactChild,
  PropsWithRef,
  ComponentProps,
  HTMLAttributes,
} from 'react'

import { InputAddon } from './InputAddon'

export type TextAreaVariantProp = 'outline' | 'flushed' | 'filled'

interface Props extends HTMLAttributes<HTMLTextAreaElement> {
  /**
   * The variant of the textarea.
   * @defaultValue outline
   */
  variant?: TextAreaVariantProp
  /**
   * Defines if the textarea is disabled.
   */
  disabled?: boolean
  /**
   * Defines if the textarea is read only.
   */
  readOnly?: boolean
  /**
   * Defines if the textarea is required.
   */
  required?: boolean
  /**
   * Defines if the textarea has error.
   */
  error?: boolean
  /**
   * Defines the type of the textarea.
   * @defaultValue text
   */
  type?: string
  /**
   * Defines if the textarea has left addon.
   */
  addonLeft?: ReactChild
  addonLeftProps?: ComponentProps<'div'>
  /**
   * Defines if the textarea has right addon.
   */
  addonRight?: ReactChild
  addonRightProps?: ComponentProps<'div'>
  /**
   * Defines if the addons are unstyled.
   */
  addonUnstyled?: boolean
  containerProps?: ComponentProps<'div'>
}

export type TextAreaProps = PropsWithRef<Props>

const TextArea = forwardRef<HTMLTextAreaElement, TextAreaProps>(
  function TextArea(
    {
      type = 'text',
      variant = 'outline',
      addonLeft,
      addonLeftProps,
      addonRight,
      addonRightProps,
      addonUnstyled,
      error,
      disabled,
      readOnly,
      required,
      className,
      children,
      containerProps,
      ...props
    },
    ref
  ) {
    const errorClasses = cn(
      'border-error-500 text-error-600 dark:border-error-400',
      variant === 'flushed' ? 'border-b rounded-t' : 'border rounded',
      addonUnstyled
        ? 'dark:text-error-300'
        : 'bg-error-50 dark:text-white dark:bg-error-600',
      { 'dark:bg-gray-800': addonUnstyled && variant === 'filled' }
    )

    const variantClasses = cn(
      'border-gray-300 text-gray-500 focus-within:shadow-primary-focus focus-within:text-primary-600 dark:text-gray-50',
      {
        'border rounded': variant === 'outline',
        'border-b rounded-t': variant === 'flushed',
        'border-0 rounded bg-gray-100 focus-within:bg-primary-100 dark:bg-gray-600 dark:focus-within:bg-primary-600':
          variant === 'filled',
      },
      (variant === 'outline' || variant === 'flushed') && {
        'bg-gray-50 focus-within:bg-primary-50 dark:bg-gray-700 dark:focus-within:bg-primary-600':
          !addonUnstyled,
        'dark:bg-gray-900 dark:focus-within:text-primary-300': addonUnstyled,
      }
    )

    const containerClasses = cn(
      'flex group transition transition-colors transition-shadow font-roboto font-sm overflow-hidden',
      disabled && 'opacity-50 dark:opacity-40',
      error ? errorClasses : variantClasses,
      containerProps?.className
    )

    const textareaClasses = cn(
      'focus:outline-none flex-1 border-0 px-3 py-2.5 text-gray-800 placeholder-gray-300 w-full dark:placeholder-gray-200 dark:text-white',
      disabled && variant !== 'filled' && 'bg-white',
      variant === 'filled'
        ? 'bg-gray-50 focus:bg-primary-50 dark:bg-gray-800'
        : 'dark:bg-gray-900 dark:focus:bg-gray-900',
      className
    )

    /**
     * Only add aria attribute to textarea if value is not falsy.
     */
    const ariaAttributes = {
      'aria-invalid': error ? true : undefined,
      'aria-required': required ? true : undefined,
      'aria-readonly': readOnly ? true : undefined,
    }

    return (
      <div className={containerClasses}>
        {addonLeft && (
          <InputAddon
            unstyled={addonUnstyled}
            hasBorder={variant === 'outline' || (variant === 'filled' && error)}
            error={error}
            disabled={disabled}
            side="left"
            {...addonLeftProps}
          >
            {addonLeft}
          </InputAddon>
        )}
        <textarea
          disabled={disabled}
          required={required}
          readOnly={readOnly}
          className={textareaClasses}
          ref={ref}
          {...ariaAttributes}
          {...props}
        />
        {addonRight && (
          <InputAddon
            unstyled={addonUnstyled}
            hasBorder={variant === 'outline' || (variant === 'filled' && error)}
            error={error}
            disabled={disabled}
            side="right"
            {...addonRightProps}
          >
            {addonRight}
          </InputAddon>
        )}
      </div>
    )
  }
)

export { TextArea }
