import tw, { css, styled } from 'twin.macro'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import type { IconName } from '@fortawesome/fontawesome-svg-core'
import type { ThemeMode } from '$types'

export interface ButtonProps {
  size?: 'xsmall' | 'small' | 'medium' | 'large'
  variant?:
    | 'solid-gray'
    | 'solid-primary'
    | 'solid-danger'
    | 'solid-success'
    | 'outline'
    | 'ghost'
    | 'link'
  mode?: ThemeMode
  container?: 'round' | 'square'
  leftIcon?: IconName
  rightIcon?: IconName
  copy?: string
  disabled?: boolean
}

const StyledButton = styled.button(
  ({
    mode = 'light',
    variant = 'solid-primary',
    size = 'medium',
    container = 'square',
  }: Partial<ButtonProps>) => [
    tw`flex items-center flex-grow-0 flex-shrink-0`,
    tw`font-bold capitalize`,
    tw`disabled:opacity-40`,
    tw`transition`,
    container === 'round' && tw`rounded-full`,
    container === 'square' && tw`rounded`,
    mode === 'dark' && colors.dark[variant],
    mode === 'light' && colors.light[variant],
    sizes[size],
    variant === 'outline' && ['small', 'xsmall'].includes(size) && tw`border-1.5`,
  ]
)

export const Button = ({ leftIcon, rightIcon, copy, ...props }: ButtonProps) => (
  <StyledButton {...props}>
    {leftIcon && <FontAwesomeIcon icon={leftIcon} />}
    {copy}
    {rightIcon && <FontAwesomeIcon icon={rightIcon} />}
  </StyledButton>
)

export const ButtonGroup = styled.div`
  ${tw`flex flex-wrap gap-2`}
`

const sizes = {
  xsmall: tw`p-2 text-xs`,
  small: tw`p-2 text-sm`,
  medium: tw`px-3 py-2`,
  large: tw`px-4 py-3 text-lg`,
}

const colors = {
  dark: {
    'solid-gray': tw`bg-gray-800 text-primary-200 hover:(bg-scrim-light)`,
    'solid-primary': tw`bg-primary-500 text-gray-0 hover:(bg-scrim-light)`,
    'solid-danger': tw`bg-danger-500 text-gray-0 hover:(bg-scrim-light)`,
    'solid-success': tw`bg-success-500 text-gray-0 hover:(bg-scrim-light)`,
    outline: tw`bg-transparent text-primary-200 border-2 border-primary-200 hover:(border-primary-500 bg-primary-500 text-gray-0)`,
    ghost: tw`bg-transparent text-primary-200 hover:(bg-primary-100 text-primary-600)`,
    link: tw`bg-transparent text-primary-200 hover:(underline)`,
  },
  light: {
    'solid-gray': tw`bg-gray-800 text-primary-200 hover:(bg-scrim-light)`,
    'solid-primary': tw`bg-primary-500 text-gray-0 hover:(bg-scrim-light)`,
    'solid-danger': tw`bg-danger-500 text-gray-0 hover:(bg-scrim-light)`,
    'solid-success': tw`bg-success-500 text-gray-0 hover:(bg-scrim-light)`,
    outline: tw`bg-transparent text-primary-600 border-2 border-primary-500 hover:(bg-primary-500 text-gray-0)`,
    ghost: tw`bg-transparent text-primary-600 hover:(bg-primary-100)`,
    link: tw`bg-transparent text-primary-600 hover:(underline)`,
  },
}
