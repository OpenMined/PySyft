import { forwardRef } from 'react'
import tw, { styled } from 'twin.macro'
import type {
  ReactElement,
  ForwardRefRenderFunction,
  InputHTMLAttributes,
  PropsWithRef,
} from 'react'

export type FormControlProps = {
  optional?: boolean
  label?: string
  hint?: string
  error?: string
} & InputProps &
  InputHTMLAttributes<HTMLInputElement>

interface InputProps {
  variant?: 'outline' | 'flushed' | 'filled'
  error?: boolean | string
  leftAddOnProps?: ReactElement
  rightAddOnProps?: ReactElement
}

export const Label = styled.label`
  ${tw`text-gray-500 text-sm font-bold`}
`

export const Input = styled.input(({ variant, error }: InputProps) => [
  variant === 'outline' && tw`border rounded p-3 items-center`,
  tw`focus:(ring-transparent outline-none ring ring-0)`,
  tw`hover:(filter)`,
  error && tw`border-red-500 focus:(shadow-focus-danger) hover:(drop-shadow-hover-danger)`,
  !error && tw`border-gray-300 focus:(shadow-focus) hover:(drop-shadow-hover)`,
  tw`focus-visible:(ring-0)`,
  tw`w-full`,
])

export const Hint = styled.div``

export const Error = styled(Hint)`
  ${tw`text-sm text-danger-600 text-right`}
`

export const FormControl = forwardRef<HTMLInputElement, PropsWithRef<FormControlProps>>(
  (props, ref) => {
    const {
      variant = 'outline',
      leftAddOnProps,
      rightAddOnProps,
      label,
      name,
      required,
      hint,
      error,
      optional,
      placeholder,
      type,
      ...rest
    } = props
    const inputProps = {
      ref,
      type,
      placeholder,
      required,
      variant,
      error,
      name,
      ...rest,
    }

    return (
      <div key={name} tw="flex flex-col gap-2">
        {label && (
          <Label htmlFor={name}>
            <span>{label}</span>
            {required && <span tw="text-primary-600"> *</span>}
            {optional && <span tw="font-normal italic text-primary-600 text-xs"> (optional)</span>}
          </Label>
        )}
        <Input {...inputProps} />
        {!error && hint && <Hint>{hint}</Hint>}
        {error && <Error>{error}</Error>}
      </div>
    )
  }
)
