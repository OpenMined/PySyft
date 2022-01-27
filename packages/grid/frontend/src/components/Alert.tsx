import tw, { styled } from 'twin.macro'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import CloseIcon from '$icons/CloseIonicons'
import type { IconName } from '@fortawesome/fontawesome-svg-core'
import type { TwStyle } from 'twin.macro'

type AlertType = 'success' | 'info' | 'warning' | 'danger'

interface AlertProps {
  alertType: AlertType
  description: string
  title: string
  variant: 'subtle' | 'solid'
  leftAccent?: boolean
  topAccent?: boolean
  centered?: boolean
  multiline?: boolean
  onClose?: () => void
}

export const Alert = ({
  variant = 'subtle',
  alertType = 'info',
  leftAccent,
  topAccent,
  onClose,
  centered,
  multiline,
  title,
  description,
}: AlertProps) => {
  const alertProps = { variant, alertType, leftAccent, topAccent, centered, multiline }
  return (
    <Container {...alertProps}>
      {multiline ? (
        <div tw="flex flex-grow gap-3 p-2">
          <FontAwesomeIcon icon={icon[alertType]} css={[iconStyle[alertType]]} tw="items-start" />
          <div tw="flex flex-col gap-0.5">
            <span tw="font-bold">{title}</span>
            <span tw="text-sm">{description}</span>
          </div>
        </div>
      ) : (
        <div tw="flex items-center gap-3 flex-grow">
          <FontAwesomeIcon icon={icon[alertType]} css={[iconStyle[alertType]]} tw="items-start" />
          <span tw="font-bold">{title}</span>
          <span tw="text-sm">{description}</span>
        </div>
      )}
      {typeof onClose === 'function' && (
        <button tw="flex-shrink-0 self-start" onClick={onClose}>
          <CloseIcon />
        </button>
      )}
    </Container>
  )
}

const Container = styled.div(
  ({ variant, alertType, leftAccent, topAccent, centered, multiline }: Partial<AlertProps>) => [
    tw`w-full flex min-h-[56px] p-2 rounded gap-2.5`,
    multiline && tw`min-h-[80px]`,
    centered && tw`text-center`,
    variant === 'solid' && solid[alertType],
    variant === 'subtle' && subtle[alertType],
    leftAccent && tw`border-l-4`,
    topAccent && tw`border-t-4`,
  ]
)

const solid: Record<AlertType, TwStyle> = {
  danger: tw`bg-danger-500 text-gray-0`,
  success: tw`bg-success-500 text-gray-0`,
  warning: tw`bg-warning-500 text-gray-0`,
  info: tw`bg-primary-500 text-gray-0`,
}

const subtle: Record<AlertType, TwStyle> = {
  danger: tw`bg-danger-100 border-danger-600 text-gray-800`,
  success: tw`bg-success-100 border-success-600 text-gray-800`,
  warning: tw`bg-warning-100 border-warning-600 text-gray-800`,
  info: tw`bg-primary-100 border-primary-600 text-gray-0`,
}

const iconStyle: Record<AlertType, TwStyle> = {
  danger: tw`text-danger-500`,
  success: tw`text-success-500`,
  warning: tw`text-warning-500`,
  info: tw`text-primary-500`,
}

const icon: Record<AlertType, IconName> = {
  danger: 'exclamation-circle',
  success: 'check-circle',
  info: 'info-circle',
  warning: 'exclamation-triangle',
}
