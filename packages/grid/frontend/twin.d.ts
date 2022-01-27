import 'twin.macro'
import styledImport from '@emotion/styled'
import { css as cssImport } from '@emotion/react'

declare module 'twin.macro' {
  const styled: typeof styledImport
  const css: typeof cssImport
}
