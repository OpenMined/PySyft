import tw, { styled } from 'twin.macro'

export interface BadgeProps {
  variant?: 'outline' | 'subtle' | 'solid'
  color?: 'primary' | 'gray' | 'danger' | 'success'
  mode?: 'light' | 'dark'
}

const outline = {
  light: {
    gray: tw`border border-gray-600 text-gray-600`,
    primary: tw`border border-primary-500 text-primary-600`,
    danger: tw`border border-danger-500 text-danger-600`,
    success: tw`border border-success-500 text-success-600`,
  },
  dark: {
    gray: tw`border border-gray-200 text-gray-200`,
    primary: tw`border border-primary-200 text-primary-200`,
    danger: tw`border border-danger-200 text-success-200`,
    success: tw`border border-success-200 text-success-200`,
  },
}

const subtle = {
  light: {
    gray: tw`bg-gray-100 text-gray-800`,
    primary: tw`bg-primary-100 text-primary-600`,
    danger: tw`bg-danger-100 text-danger-600`,
    success: tw`bg-success-100 text-success-600`,
  },
  dark: {
    gray: tw`bg-gray-100 text-gray-800`,
    primary: tw`bg-primary-100 text-primary-600`,
    danger: tw`bg-danger-100 text-danger-600`,
    success: tw`bg-success-100 text-success-600`,
  },
}

const solid = {
  light: {
    gray: tw`bg-gray-800 text-primary-200`,
    primary: tw`bg-primary-500 text-gray-0`,
    danger: tw`bg-danger-500 text-gray-0`,
    success: tw`bg-success-500 text-gray-0`,
  },
  dark: {
    gray: tw`bg-gray-800 text-primary-200`,
    primary: tw`bg-primary-500 text-gray-0`,
    danger: tw`bg-danger-500 text-gray-0`,
    success: tw`bg-success-500 text-gray-0`,
  },
}

const BaseBadge = styled.div<BadgeProps>`
  ${tw`inline rounded-sm leading-normal text-xs font-bold truncate overflow-ellipsis px-[6px] py-[2px]`}
`

export const Badge = styled(BaseBadge)(
  ({ variant = 'subtle', color = 'gray', mode = 'light' }: BadgeProps) => [
    variant === 'subtle' && subtle[mode]?.[color],
    variant === 'outline' && outline[mode]?.[color],
    variant === 'solid' && solid[mode]?.[color],
  ]
)
