import React from 'react'
import cn from 'classnames'

export type AvatarProps = {
  size?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | '3xl'
  variant?: 'primary' | 'error' | 'warning' | 'gray' | 'success'
  show?: boolean
  className?: string
} & React.ComponentProps<'img'>

const avatarSizes = {
  sm: 'w-6 h-6',
  md: 'w-8 h-8',
  lg: 'w-10 h-10',
  xl: 'w-12 h-12',
  '2xl': 'w-16 h-16',
  '3xl': 'w-20 h-20',
}

const alertSizes = {
  sm: 'w-2 h-2',
  md: 'w-2.5 h-2.5',
  lg: 'w-3 h-3',
  xl: 'w-3.5 h-3.5',
  '2xl': 'w-4 h-4',
  '3xl': 'w-6 h-6',
}

export function Avatar({
  size = 'md',
  variant = 'primary',
  show,
  className,
  ...props
}: AvatarProps) {
  const classes = cn(avatarSizes[size], 'rounded-full', className)
  const indicatorClasses = cn(
    !show && 'hidden',
    'absolute bottom-0 right-0 block rounded-full ring-2 ring-white bg-gradient-to-r from-gradient-white',
    `bg-${variant}-500`,
    alertSizes[size]
  )

  return (
    <span className="inline-block relative">
      <img className={classes} {...props} style={{ maxWidth: '48px' }} />
      <span className={indicatorClasses} />
    </span>
  )
}
