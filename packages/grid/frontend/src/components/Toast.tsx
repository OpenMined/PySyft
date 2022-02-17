import tw, { styled } from 'twin.macro'
import toast, { useToaster } from 'react-hot-toast'
import CloseIcon from '$icons/CloseIonicons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { IconName } from '@fortawesome/free-solid-svg-icons'

type ToastType = 'success' | 'danger' | 'info' | 'warning'

type ToastVariant = 'subtle' | 'solid' | 'dark' | 'accent'

export type ExtendedToastProps = {
  id: string
  toastType: ToastType
  variant: ToastVariant
  title: string
} & ReturnType<typeof toast>

export const Toast = () => {
  const { toasts } = useToaster()

  return (
    <div tw="top-2 right-2 z-40 fixed flex flex-col gap-3">
      {toasts.map(t => {
        const {
          id,
          title,
          message,
          visible,
          variant = 'subtle',
          toastType: type = 'info',
        } = t as typeof t & ExtendedToastProps
        const containerProps = { variant, type, visible }
        return (
          <ToastContainer
            key={`toast-${id}`}
            role="alert"
            aria-label="notification"
            data-cy="toast"
            {...containerProps}
          >
            <div tw="p-2 flex-grow flex items-start gap-3">
              <FontAwesomeIcon icon={icons[type]} css={[variants[type][variant].icon]} />
              <div tw="gap-0.5">
                <span tw="block font-bold capitalize">{title}</span>
                <span tw="block">{message}</span>
              </div>
            </div>
            <button
              type="button"
              tw="self-start"
              onClick={() => toast.dismiss(id)}
              css={variants[type][variant].icon}
            >
              <CloseIcon />
            </button>
          </ToastContainer>
        )
      })}
    </div>
  )
}

interface ToastContainerProps {
  variant: 'subtle' | 'solid' | 'dark' | 'accent'
  type: 'info' | 'success' | 'warning' | 'danger'
  visible: boolean
}

const ToastContainer = styled.div(({ visible, variant, type }: ToastContainerProps) => [
  tw`flex min-w-[320px] w-full p-2 rounded justify-between transition-all duration-1000 ease-in-out gap-2.5 bg-gray-100`,
  variants[type][variant].box,
  visible ? tw`opacity-100` : tw`opacity-0`,
])

// its really long
const info = {
  solid: {
    box: tw`bg-primary-500 text-gray-0`,
    icon: tw`text-gray-0`,
    close: tw`text-gray-0`,
  },
  subtle: {
    box: tw`bg-primary-100 text-gray-800`,
    icon: tw`text-primary-600`,
    close: tw`text-primary-600`,
  },
  accent: {
    box: tw`border-primary-500 text-gray-800`,
    icon: tw`text-primary-500`,
    close: tw`text-primary-500`,
  },
  dark: {
    box: tw`bg-gray-800 text-gray-0`,
    icon: tw`text-primary-200`,
    close: tw`text-primary-200`,
  },
}

const success = {
  solid: {
    box: tw`bg-success-500 text-gray-0`,
    icon: tw`text-gray-0`,
    close: tw`text-gray-0`,
  },
  subtle: {
    box: tw`bg-success-100 text-gray-800`,
    icon: tw`text-success-600`,
    close: tw`text-success-600`,
  },
  accent: {
    box: tw`bg-gray-0 border-t-4 border-success-500 text-gray-800`,
    icon: tw`text-success-500`,
    close: tw`text-success-500`,
  },
  dark: {
    box: tw`bg-gray-800 text-gray-0`,
    icon: tw`text-success-200`,
    close: tw`text-success-200`,
  },
}

const warning = {
  solid: {
    box: tw`bg-warning-500 text-gray-0`,
    icon: tw`text-gray-0`,
    close: tw`text-gray-0`,
  },
  subtle: {
    box: tw`bg-warning-100 text-gray-800`,
    icon: tw`text-warning-600`,
    close: tw`text-warning-600`,
  },
  accent: {
    box: tw`border-warning-500 text-gray-800`,
    icon: tw`text-warning-500`,
    close: tw`text-warning-500`,
  },
  dark: {
    box: tw`bg-gray-800 text-gray-0`,
    icon: tw`text-warning-200`,
    close: tw`text-warning-200`,
  },
}

const danger = {
  solid: {
    box: tw`bg-danger-500 text-gray-0`,
    icon: tw`text-gray-0`,
    close: tw`text-gray-0`,
  },
  subtle: {
    box: tw`bg-danger-100 text-gray-800`,
    icon: tw`text-danger-600`,
    close: tw`text-danger-600`,
  },
  accent: {
    box: tw`border-danger-500 text-gray-800`,
    icon: tw`text-danger-500`,
    close: tw`text-danger-500`,
  },
  dark: {
    box: tw`bg-gray-800 text-gray-0`,
    icon: tw`text-danger-200`,
    close: tw`text-danger-200`,
  },
}

const variants = {
  info,
  success,
  warning,
  danger,
}

const icons: Record<ToastType, IconName> = {
  success: 'check-circle',
  danger: 'exclamation-circle',
  info: 'info-circle',
  warning: 'exclamation-triangle',
}
