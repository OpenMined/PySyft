import React from 'react'
import cn from 'classnames'
import { Text } from '../Typography/Text'
import { Icon } from '../Icon/Icon'
import type { ElementType, MouseEventHandler, PropsWithChildren } from 'React'
import type { OmuiColors } from '@/styles/colorType'

export type TagSizeProp = 'sm' | 'md' | 'lg'
export type TagTypeProp = 'round' | 'square'
export type TagVariantProp = 'gray' | 'primary' | 'tertiary' | 'quaternary'

const VARIANT_COLORS: Record<TagVariantProp, Partial<OmuiColors>> = {
  gray: 'gray',
  primary: 'primary',
  tertiary: 'violet',
  quaternary: 'blue',
} as const

interface Props {
  /**
   * The variant of the Tag.
   * @defaultValue outline
   */
  variant?: TagVariantProp
  /**
   * The size of the Tag.
   * @defaultValue sm
   */
  size?: TagSizeProp
  /**
   * Defines if the tag should be disabled.
   * @defaultValue false
   */
  disabled?: boolean
  /**
   * The style of the Tag. The round style is similar to the artist formerly known as pill.
   * @defaultValue square
   */
  tagType?: TagTypeProp
  /**
   * Classes that are passed down to the main component (button or span).
   */
  className?: string
  /**
   * When an onClick function is set, the tag becomes a button
   */
  onClick?: MouseEventHandler<HTMLButtonElement>
}

interface PropsWithoutIcons extends Props {
  icon: null | undefined
  iconSide: never
}

export type TagIconSideProp = 'left' | 'right'

interface PropsWithIcons extends Props {
  /**
   * Icon component to be rendered next to the Tag.
   */
  icon: ElementType
  /**
   * Position of the icon, left or right of the inner tag text. It is not allowed to have both positions or multiple icons.
   */
  iconSide: TagIconSideProp
}

export type TagProps = PropsWithChildren<PropsWithoutIcons | PropsWithIcons>

const tagTypeStyles: Record<TagTypeProp, string> = {
  round: 'rounded-full',
  square: 'rounded-sm',
}

const tagBackgoundAndTextColor: Record<TagVariantProp, string> = Object.assign(
  {},
  ...(Object.keys(VARIANT_COLORS) as TagVariantProp[]).map(variant => {
    if (variant === 'gray') {
      return { gray: 'text-gray-600 bg-gray-100 hover:text-primary-200 hover:bg-gray-800' }
    }
    const color = VARIANT_COLORS[variant]
    return { [variant]: `text-${color}-600 bg-${color}-100 hover:text-white hover:bg-${color}-500` }
  })
)

const defaultClasses = 'inline-flex items-center px-2.5 py-1 transition transition-colors'

/**
 * Tags are meant to indicate categories and typically links out to filtering mechanism.
 */
const Tag = ({
  variant = 'primary',
  size = 'md',
  tagType = 'square',
  disabled,
  icon,
  iconSide,
  className,
  children,
  ...props
}: TagProps) => {
  const clickable = typeof props.onClick === 'function'
  const Component: keyof JSX.IntrinsicElements = clickable ? 'button' : 'span'
  const classes = cn(
    defaultClasses,
    tagBackgoundAndTextColor[variant],
    tagTypeStyles[tagType],
    clickable ? 'cursor-pointer' : 'cursor-default',
    disabled && 'pointer-events-none opacity-50',
    className
  )
  const iconClasses = cn(iconSide === 'right' && 'ml-1.5', iconSide === 'left' && 'mr-1.5')

  return (
    <Component {...props} type="button" className={classes}>
      {icon && iconSide === 'left' ? (
        <Icon icon={icon} size={size} variant="ghost" className={iconClasses} />
      ) : null}
      <Text size={size}>{children}</Text>
      {icon && iconSide === 'right' ? (
        <Icon icon={icon} size={size} variant="ghost" className={iconClasses} />
      ) : null}
    </Component>
  )
}

export { Tag }
