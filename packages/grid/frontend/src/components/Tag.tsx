import tw, { styled } from 'twin.macro'

interface TagProps {
  style?: 'rounder' | 'square'
  size?: 'small' | 'medium' | 'large'
  color?: 'primary' | 'gray'
  disabled?: boolean
}

export const Tag = styled.div(
  ({ style = 'rounder', size = 'medium', color = 'gray', disabled = false }: TagProps) => [
    tw`flex items-center py-1 px-2 leading-[1.5] cursor-default`,
    style === 'rounder' && tw`rounded-full`,
    style === 'square' && tw`rounded-md`,
    size === 'small' && tw`text-sm`,
    size === 'medium' && tw`text-base`,
    size === 'large' && tw`text-lg`,
    color === 'gray' && tw`bg-gray-100 text-gray-600 hover:(bg-gray-800 text-primary-200)`,
    color === 'primary' && tw`bg-primary-100 text-primary-800 hover:(text-gray-0 bg-primary-500)`,
    disabled && tw`opacity-50`,
  ]
)
