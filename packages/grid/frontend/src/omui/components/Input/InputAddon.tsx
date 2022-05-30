import React from 'react'
import cn from 'classnames'
import type { HTMLAttributes, PropsWithChildren } from 'react'

interface Props extends HTMLAttributes<HTMLDivElement> {
  unstyled?: boolean
  hasBorder?: boolean
  error?: boolean
  disabled?: boolean
  side: 'left' | 'right'
}

export type InputAddonProps = PropsWithChildren<Props>

const InputAddon = ({
  unstyled,
  hasBorder,
  error,
  disabled,
  children,
  side,
  ...props
}: InputAddonProps) => {
  const addonClasses = cn(
    'flex items-center justify-center px-3',
    error ? 'border-error-500' : 'border-gray-300',
    hasBorder &&
      !unstyled && {
        'border-r': side === 'left',
        'border-l': side === 'right',
      },
    props.onClick && 'cursor-pointer'
  )

  return (
    <div {...props} className={addonClasses}>
      {children}
    </div>
  )
}

export { InputAddon }
