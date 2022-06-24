import React, { forwardRef } from 'react'
import cn from 'classnames'
import type {
  ButtonHTMLAttributes,
  ElementType,
  PropsWithChildren,
} from 'react'
import { Text } from '../Typography/Text'
import { Icon, IconSizeProp } from '../Icon/Icon'
import { NewSpinner } from '@/components/NewSpinner'

export type ButtonSize = 'xs' | 'sm' | 'md' | 'lg'
export type ButtonVariant = 'gray' | 'primary' | 'outline' | 'ghost' | 'link'
export type ButtonColor = 'gray' | 'primary' | 'warning' | 'error' | 'success'

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  /**
   * The size of the Button.
   * @defaultValue md
   */
  size?: ButtonSize
  /**
   * The variant of the Button.
   * @defaultValue primary
   */
  variant?: ButtonVariant
  color?: ButtonColor
  leftIcon?: ElementType
  rightIcon?: ElementType
  isLoading?: boolean
}

export type ButtonProps = PropsWithChildren<Props>

type iconSizeRecord<T> = Record<ButtonSize, T>

/**
 * To keep the same size I added this logic:
 * When the button have border, it should have a smaller padding;
 * The problem is with box-sizing, I could not work around it.
 *
 * Other solution would be add border transparent to all variants and keep the same padding:
 * The problem found with this solution was that the button glitch on gradient effect from hover.
 */
const buttonBorderlessSize: iconSizeRecord<string> = {
  xs: 'p-2',
  sm: 'p-2',
  md: 'px-3 py-2',
  lg: 'px-4 py-3',
}
const buttonBorderSizes: iconSizeRecord<string> = {
  xs: 'p-1.5',
  sm: 'p-1.5',
  md: 'px-2.5 py-1.5',
  lg: 'px-3.5 py-2.5',
}
const iconSizes: iconSizeRecord<IconSizeProp> = {
  xs: 'md',
  sm: 'md',
  md: 'lg',
  lg: 'xl',
}

const variants = (color: string): Record<ButtonVariant, string> => ({
  gray: 'bg-gray-800 text-primary-200 hover:from-gradient-white bg-gradient-to-r dark:hover:from-gbsc',
  primary: `bg-${color}-500 text-white hover:from-gradient-white bg-gradient-to-r`,
  outline: `bg-transparent text-${color}-600 border-2 border-${color}-500 hover:bg-${color}-500 hover:text-white dark:text-${color}-200 dark:border-${color}-200 dark:hover:border-${color}-500 dark:hover:text-white`,
  ghost: `text-${color}-600 dark:text-${color}-200 hover:bg-${color}-100 dark:hover:text-${color}-600`,
  link: 'text-primary-600 hover:underline dark:text-primary-200',
})

const defaultClasses = 'inline-flex items-center rounded gap-x-1.5 outline-none'
const disabledClasses = 'opacity-40 pointer-events-none'

const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    size = 'md',
    variant = 'primary',
    color = 'primary',
    disabled,
    leftIcon,
    rightIcon,
    isLoading = false,
    className,
    children,
    ...props
  },
  ref
) {
  const hasBorder = variant === 'outline'
  const buttonClasses = cn(
    defaultClasses,
    hasBorder ? buttonBorderSizes[size] : buttonBorderlessSize[size],
    variants(color)[variant],
    (isLoading || disabled) && disabledClasses,
    className
  )
  return (
    <button
      className={buttonClasses}
      ref={ref}
      disabled={isLoading || disabled}
      {...props}
    >
      {leftIcon && (
        <Icon
          size={iconSizes[size]}
          variant="ghost"
          icon={leftIcon}
          containerProps={{ className: 'pr-2' }}
        />
      )}

      <NewSpinner className={cn({ hidden: !isLoading })} />
      {typeof children === 'string' ? (
        <Text as="span" size={size} bold>
          {children}
        </Text>
      ) : (
        children
      )}

      {rightIcon && (
        <Icon
          size={iconSizes[size]}
          icon={rightIcon}
          variant="ghost"
          containerProps={{ className: 'ml-2' }}
        />
      )}
    </button>
  )
})

export type IconButtonVariant = Exclude<ButtonVariant, 'link' | 'ghost'>

export type IconButtonProps = Exclude<
  ButtonProps,
  'rightIcon' | 'leftIcon' | 'children' | 'variant'
> & {
  icon: ElementType
  variant?: IconButtonVariant
  rounded?: boolean
}

const iconBorderlessSize: iconSizeRecord<string> = {
  xs: 'p-2',
  sm: 'p-2.5',
  md: 'p-3',
  lg: 'p-3.5',
}
const iconBorderSizes: iconSizeRecord<string> = {
  xs: 'p-1.5',
  sm: 'p-2',
  md: 'p-2.5',
  lg: 'p-3',
}

const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  function IconButton(
    {
      size = 'md',
      variant = 'primary',
      disabled,
      icon,
      rounded,
      className,
      ...props
    },
    ref
  ) {
    const hasBorder = variant === 'outline'
    const iconButtonClasses = cn(
      defaultClasses,
      hasBorder ? iconBorderSizes[size] : iconBorderlessSize[size],
      variants[variant],
      disabled && disabledClasses,
      rounded && 'rounded-full',
      className
    )
    return (
      <button
        className={iconButtonClasses}
        disabled={disabled}
        ref={ref}
        {...props}
      >
        <Icon size={iconSizes[size]} icon={icon} variant="ghost" />
      </button>
    )
  }
)

export { Button, IconButton }
