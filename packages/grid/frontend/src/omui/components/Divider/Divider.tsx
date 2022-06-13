import React from 'react'
import cn from 'classnames'
import type { HTMLAttributes } from 'react'

export type DividerOrientation = 'horizontal' | 'vertical'
export type DividerColor = 'light' | 'black' | 'dark'
export type DividerProps = {
  /**
   * Sets the divider orientation
   * @defaultValue horizontal
   */
  orientation?: DividerOrientation
  /**
   * Sets the divider theme to black (gray/900), light (gray/200) or dark (gray/700)
   * @defaultValue dark
   */
  color?: DividerColor
  /**
   * Classes that are passed to the hr element
   */
  className?: string
}

type Themes = {
  [k in DividerColor]: string | string[]
}
const dividerColorClasses: Themes = {
  light: 'border-gray-200',
  dark: 'border-gray-700',
  black: 'border-gray-900',
}

type Orientation = {
  [k in DividerOrientation]: string | string[]
}
const orientationClasses: Orientation = {
  horizontal: 'border-t w-full my-2',
  vertical: 'border-l h-auto min-h-full w-px',
}

export function Divider({
  orientation = 'horizontal',
  color = 'dark',
  className,
  ...props
}: DividerProps & HTMLAttributes<HTMLHRElement>) {
  const classes = cn(
    dividerColorClasses[color],
    orientationClasses[orientation],
    className
  )
  return (
    <hr
      aria-orientation={orientation}
      role="separator"
      className={classes}
      {...props}
    />
  )
}
