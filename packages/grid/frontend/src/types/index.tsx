import type { FormControlProps } from '$components/FormControl'
import type { RegisterOptions } from 'react-hook-form'

export type ThemeMode = 'light' | 'dark'

export type RHFElements = FormControlProps & { registerOptions?: RegisterOptions }
