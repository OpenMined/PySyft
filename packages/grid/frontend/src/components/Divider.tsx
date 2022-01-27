import tw, { styled } from 'twin.macro'
import type { ThemeMode } from '$types'

interface DividerProps {
  mode?: ThemeMode
}

export const Divider = styled.hr(({ mode }: DividerProps) => [
  tw`border my-1`,
  mode === 'light' && tw`border-gray-200`,
  mode === 'dark' && tw`border-gray-700`,
])
