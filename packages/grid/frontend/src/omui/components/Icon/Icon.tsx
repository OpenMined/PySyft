import React from 'react'
import cn from 'classnames'
import type { ElementType, ComponentPropsWithoutRef } from 'react'

export type IconSizeProp = 'xs' | 'sm' | 'md' | 'lg' | 'xl'
export type IconVariantProp = 'gray' | 'solid' | 'subtle' | 'outline' | 'ghost'
export type IconContainerProp = 'round' | 'square'

export interface IconProps {
  /**
   * The size of the icon.
   * @defaultValue md
   */
  size?: IconSizeProp
  /**
   * The variant of the icon.
   * @defaultValue solid
   */
  variant?: IconVariantProp
  /**
   * The container of the icon
   * @defaultValue round
   */
  container?: IconContainerProp
  /**
   * The React Component containing the SVG icon
   * @defaultValue null
   */
  icon: ElementType
  /**
   * Props that are passed to the <span> component placed before the icon
   */
  containerProps?: ComponentPropsWithoutRef<'span'>
  /**
   * Optional classes that are passed to the Icon component
   */
  className?: string
  /** Icon title for accessibility purposes. Only to be used if the icon is a standalone, meaningful
   * icon. If you're using SVGs, please add a <g aria-hidden> wrapper or similar to the element. The
   * icon element will receive the following attributes: aria-hidden="false" role="img"
   * focusable="false" and aria-label={title}. All attributes can be overwritten by passing them as
   * props to the Icon component.
   */
  title?: string
}

type Variants = {
  [k in IconVariantProp]: string | string[]
}

const variants: Variants = {
  outline: [
    'bg-transparent shadow-icon-border',
    'border-primary-500 border dark:border-primary-200 text-primary-500 dark:text-primary-200',
  ],
  gray: 'bg-gray-800 text-primary-200',
  solid: 'bg-primary-500 text-white',
  subtle: 'bg-primary-200 text-primary-600',
  ghost: 'text-current',
}

type Sizes = {
  [k in 'container' | 'icon']: {
    [o in IconSizeProp]: string | string[]
  }
}

const sizes: Sizes = {
  container: {
    xs: 'w-6 h-6',
    sm: 'w-8 h-8',
    md: 'w-9 h-9',
    lg: 'w-11 h-11',
    xl: 'w-12 h-12',
  },
  icon: {
    xs: 'w-3 h-3',
    sm: 'w-3.5 h-3.5',
    md: 'w-4 h-4',
    lg: 'w-4.5 h-4.5',
    xl: 'w-5 h-5',
  },
}

type Borders = {
  [k in IconContainerProp]: string
}

const borders: Borders = {
  round: 'rounded-full',
  square: 'rounded-md',
}

const defaultClasses = 'inline-flex items-center justify-center'

const Icon = React.forwardRef<HTMLSpanElement, IconProps>(function Icon(
  {
    size = 'md',
    variant = 'solid',
    container = 'round',
    icon: IconElement,
    className,
    containerProps,
    title,
    ...props
  },
  ref
) {
  const containerClasses = cn(
    defaultClasses,
    variants[variant],
    borders[container],
    variant !== 'ghost' && sizes.container[size],
    containerProps?.className
  )
  const iconClasses = cn(sizes.icon[size], className)

  if (variant === 'ghost') {
    return (
      <span ref={ref}>
        <IconElement
          className={iconClasses}
          focusable={false}
          aria-hidden={title ? undefined : true}
          aria-label={title}
          role={title ? 'img' : 'presentation'}
          {...props}
        />
      </span>
    )
  }

  return (
    <span {...containerProps} className={containerClasses} ref={ref}>
      <IconElement
        className={iconClasses}
        focusable={false}
        aria-hidden={title ? undefined : true}
        aria-label={title}
        role={title ? 'img' : 'presentation'}
        {...props}
      />
    </span>
  )
})

export { Icon }
