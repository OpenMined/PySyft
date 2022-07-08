import React, { forwardRef } from 'react'
import type { PropsWithRef, ReactNode } from 'react'
import cn from 'classnames'
import { Text } from '../Typography/Text'
import { Optional } from '@/components/lib'

export interface Props {
  /**
   * The id of FormControl, this prop will be passed to children Input.
   */
  id: string
  /**
   * The Label of the FormControl.
   */
  label?: string
  /**
   * The Helper Text of the FormControl.
   */
  hint?: ReactNode
  /**
   * If the field is optional
   */
  optional?: boolean
  /**
   * Determine the state of FormControl's children Input, Label and Helper Text as disabled.
   */
  disabled?: boolean
  /**
   * Determine the state of FormControl's children Input and Label as required.
   */
  required?: boolean
  /**
   * Determine the state of FormControl's children Input, Label and Helper Text as invalid.
   */
  error?: boolean
  children: ReactNode
  className?: string
}

export type FormControlProps = PropsWithRef<Props>

const FormControl = forwardRef<HTMLDivElement, FormControlProps>(
  function FormControl(
    {
      label,
      hint,
      disabled,
      required,
      error,
      optional,
      id,
      className,
      children,
      ...props
    },
    ref
  ) {
    const hintId = `omui-form-control-${id}`
    const textStates = cn(
      'text-gray-500 dark:text-gray-200',
      disabled && 'opacity-50 pointer-events-none',
      error &&
        'text-error-600 dark:text-error-200 fill-error-200 dark:fill-error-200'
    )
    const requiredClasses = cn(
      'ml-1',
      !error && 'text-primary-500 dark:text-primary-400'
    )
    const hintClasses = cn('mt-2 px-2', textStates)

    const childProps = {
      required,
      disabled,
      error,
      id,
      'aria-describedby': hint ? hintId : undefined,
    }

    return (
      <div className={cn('w-full flex flex-col', className)} {...props}>
        {label && (
          <label htmlFor={id} className="mb-2">
            <Text bold size="sm" className="text-gray-500 capitalize">
              {label}
            </Text>
            {/* TODO: sub below with the Required icon... or maybe superscript? */}
            {required && (
              <Text size="sm" className={requiredClasses} aria-hidden="true">
                {' '}
                *
              </Text>
            )}
            {optional && <Optional size="xs" />}
          </label>
        )}
        {React.isValidElement(children)
          ? React.cloneElement(children, childProps)
          : children}
        {hint && (
          // TODO: Update Text to extend HTMLElement intending to accept id, title, etc.
          <Text
            id={hintId}
            size="sm"
            className={hintClasses}
            aria-live={error ? 'polite' : undefined}
          >
            {hint}
          </Text>
        )}
      </div>
    )
  }
)

export { FormControl }
