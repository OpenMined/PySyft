import React from 'react'
import cn from 'classnames'
import type { PropsWithChildren } from 'react'

export type TextAsProp = 'p' | 'span' | 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6'
export type TextSizeProp =
  | 'xs'
  | 'sm'
  | 'md'
  | 'lg'
  | 'xl'
  | '2xl'
  | '3xl'
  | '4xl'
  | '5xl'
  | '6xl'

interface Props {
  /**
   * The HTML element of the component.
   * @defaultValue p
   *
   */
  as?: TextAsProp
  /**
   * The font size of the text.
   * @defaultValue md
   *
   */
  size?: TextSizeProp
  /**
   * Defines if the text should be bold.
   */
  bold?: boolean
  className?: string
}

interface TextMonoProps {
  /**
   * Defines if the font family should be monospaced.
   */
  mono?: true
  underline?: never
  uppercase?: never
}

interface TextRegularProps {
  mono?: false
  /**
   * Defines if the text should be underlined.
   */
  underline?: boolean
  /**
   * Defines if the text should be uppercased.
   */
  uppercase?: boolean
}
/**
 * Wrapper for Text default props.
 * @typeParam T - Props that do not have discriminated union.
 */
type DefaultProps<T> = PropsWithChildren<T & (TextMonoProps | TextRegularProps)>

export type TextProps = DefaultProps<Props>

export function Text({
  as = 'span',
  size = 'md',
  bold,
  uppercase,
  underline,
  mono,
  className,
  children,
  ...props
}: TextProps) {
  const isRoboto = React.useMemo(
    () => ['xs', 'sm', 'md', 'lg'].includes(size),
    [size]
  )

  const underlined = { underline: underline && !mono }
  const uppercased = { uppercase: uppercase || mono }
  const fontFamily = {
    'font-roboto': isRoboto && !mono,
    'font-rubik': !isRoboto && !mono,
    'font-firacode': mono,
  }
  const fontWeight = {
    'font-normal': isRoboto && !bold,
    'font-medium': !isRoboto && !bold,
    'font-bold': (isRoboto && bold) || (mono && bold),
    'font-black': !isRoboto && bold && !mono,
  }

  const uppercaseCustomRule =
    uppercase && ['4xl', '5xl', '6xl'].includes(size) && !mono
  const monoCustomRule = mono && ['4xl', '5xl'].includes(size)

  const fontSize = {
    [`text-${size}`]: !uppercaseCustomRule && !monoCustomRule,
    [`text-${size}-upper`]: uppercaseCustomRule,
    [`text-${size}-mono`]: monoCustomRule,
  }

  const classes = cn(
    fontFamily,
    fontWeight,
    fontSize,
    underlined,
    uppercased,
    className
  )

  return React.createElement(as, { className: classes, ...props }, children)
}

export type HeadingProps = DefaultProps<Omit<Props, 'as' | 'size'>>

export function H1(props: HeadingProps) {
  return <Text {...props} as="h1" size="5xl" />
}

export function H2(props: HeadingProps) {
  return <Text {...props} as="h2" size="4xl" />
}

export function H3(props: HeadingProps) {
  return <Text {...props} as="h3" size="3xl" />
}

export function H4(props: HeadingProps) {
  return <Text {...props} as="h4" size="2xl" />
}

export function H5(props: HeadingProps) {
  return <Text {...props} as="h5" size="xl" />
}

export function H6(props: HeadingProps) {
  return <Text {...props} as="h6" size="sm" />
}
