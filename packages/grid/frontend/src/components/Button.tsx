import { forwardRef } from 'react'
import cn from 'classnames'
import { Spinner } from '@/components'
import type { ComponentPropsWithRef } from 'react'

export function Button(props: ComponentPropsWithRef<'button'>) {
  return (
    <button
      {...props}
      className={cn(
        'flex items-center px-4 py-1 bg-cyan-800 hover:bg-opacity-70 active:bg-opacity-100 rounded-md text-white',
        props.className
      )}
    >
      {props.children}
    </button>
  )
}

interface GridButton extends ComponentPropsWithRef<'button'> {
  isLoading?: boolean
  buttonStyle?: string
}

const buttonStyles = {
  base: cn(
    'uppercase text-sm font-medium tracking-tight',
    'rounded-md items-center shadow-sm px-4 py-2',
    'border border-gray-300',
    'disabled:bg-gray-300 disabled:text-gray-800 disabled:cursor-not-allowed',
    'flex-shrink-0'
  ),
  normal: cn(
    'bg-white',
    'hover:bg-sky-500 hover:text-white',
    'active:bg-sky-800 active:text-white'
  ),
  remove: cn(
    'bg-red-700 text-gray-50',
    'hover:bg-red-500 hover:bg-opacity-100',
    'active:bg-red-800'
  ),
}

export const NormalButton = forwardRef<HTMLButtonElement, GridButton>(function NormalButton(
  props,
  ref
) {
  const { isLoading, buttonStyle = 'normal', ...buttonProps } = props
  return (
    <button
      {...buttonProps}
      ref={ref}
      className={cn(buttonStyles.base, buttonStyles[buttonStyle], props.className)}
    >
      {isLoading ? <Spinner className="w-4 h-4" /> : props.children}
    </button>
  )
})

export function DeleteButton(props: GridButton) {
  const { className, ...buttonProps } = props
  return (
    <NormalButton {...buttonProps} buttonStyle="remove" className={cn('mt-auto', className)}>
      {props.children}
    </NormalButton>
  )
}

export function ActionButton(props: GridButton) {
  return (
    <NormalButton
      {...props}
      className="flex-shrink-0 w-24 mt-auto mr-4 bg-gray-700 text-gray-50 bg-opacity-80 hover:bg-opacity-100"
    />
  )
}

export function ClosePanelButton(props: GridButton) {
  return <NormalButton {...props} type="button" className="flex-shrink-0 mt-auto" />
}
