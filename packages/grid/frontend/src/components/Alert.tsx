import React, { createContext, useContext } from 'react'
import cn from 'classnames'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  faExclamationCircle,
  faExclamationTriangle,
  faInfoCircle,
  faCheckCircle,
  faTimes,
} from '@fortawesome/free-solid-svg-icons'
import { Text } from '@/omui'

import type { ReactNode } from 'react'
import type { IconDefinition } from '@fortawesome/fontawesome-svg-core'
import type { OmuiColors } from '@/omui/styles/colorType'

type AlertIconType = 'error' | 'warning' | 'info' | 'success'
type AlertAlignType = 'left' | 'center'
type AlertVariantType = 'oneline' | 'multiline'
type AlertStyleType = 'subtle' | 'solid' | 'leftAccent' | 'topAccent'

// fa-icons
const alertIcons: Record<AlertIconType | 'close', IconDefinition> = {
  error: faExclamationCircle,
  warning: faExclamationTriangle,
  info: faInfoCircle,
  success: faCheckCircle,
  close: faTimes,
}

const twAlertIconSize = 'text-xl'

export interface AlertProps {
  type?: AlertIconType
  alertStyle: AlertStyleType
  variant?: AlertVariantType
  align?: AlertAlignType
  close?: boolean
  title?: string
  description?: ReactNode
  className?: string
}

const AlertContext = createContext<AlertProps>({
  type: 'info',
  variant: 'oneline',
  align: 'left',
  alertStyle: 'subtle',
  close: false,
  title: '',
  description: '',
})

const styles = (color: OmuiColors) => ({
  subtle: `bg-${color}-100`,
  solid: `bg-${color}-500 text-white`,
  topAccent: `bg-${color}-100 border-t-4 border-${color}-500 text-gray-800`,
  leftAccent: `bg-${color}-100 border-l-4 border-${color}-500 text-gray-800`,
})

function AlertBase({
  type = 'info',
  variant = 'oneline',
  align = 'left',
  alertStyle = 'subtle',
  close = false,
  title,
  description,
  className,
}: AlertProps) {
  const vertAlign = variant === 'oneline' && 'items-center'
  return (
    <AlertContext.Provider
      value={{ type, variant, align, alertStyle, close, title, description }}
    >
      <div
        className={cn('flex justify-between px-3 py-2', vertAlign, className)}
      >
        <div className="p-2">
          {variant === 'oneline' && <AlertOneLine />}
          {variant === 'multiline' && <AlertMultiLine />}
        </div>
        {close && (
          <div className="cursor-pointer">
            <AlertIcon type="close" />
          </div>
        )}
      </div>
    </AlertContext.Provider>
  )
}

function AlertOneLine() {
  const { title, type, description, variant, align } = useContext(AlertContext)
  const vertAlign = variant === 'oneline' && 'items-center'
  const horzAlign = {
    'justify-start': align === 'left',
    'justify-center': align === 'center',
  }
  return (
    <div className={cn('flex space-x-3', vertAlign, horzAlign)}>
      <AlertIcon type={type} />
      {title && <Text bold>{title}</Text>}
      {description && React.isValidElement(description) ? (
        React.cloneElement(description)
      ) : (
        <Text>{description}</Text>
      )}
    </div>
  )
}

function AlertMultiLine() {
  const { title, type, description } = useContext(AlertContext)
  return (
    <div className="flex space-x-3 items-start">
      <AlertIcon type={type} />
      <div className="flex-col">
        {title && <Text bold>{title}</Text>}
        {description && <Text>{description}</Text>}
      </div>
    </div>
  )
}

function AlertIcon({ type }: { type: AlertIconType | 'close' }) {
  return (
    <div className="w-6 h-6 flex-shrink-0 text-current">
      <FontAwesomeIcon icon={alertIcons[type]} className={twAlertIconSize} />
    </div>
  )
}

function AlertError(props: AlertProps) {
  return (
    <AlertBase
      {...props}
      type="error"
      className={styles('error')[props.alertStyle]}
    />
  )
}

function AlertWarning(props: AlertProps) {
  return (
    <AlertBase
      {...props}
      type="warning"
      className={styles('warning')[props.alertStyle]}
    />
  )
}

function AlertInfo(props: AlertProps) {
  return (
    <AlertBase
      {...props}
      type="info"
      className={styles('primary')[props.alertStyle]}
    />
  )
}

function AlertSuccess(props: AlertProps) {
  return (
    <AlertBase
      {...props}
      type="success"
      className={styles('success')[props.alertStyle]}
    />
  )
}

export const Alert = Object.assign(
  {},
  {
    Base: AlertBase,
    Error: AlertError,
    Warning: AlertWarning,
    Info: AlertInfo,
    Success: AlertSuccess,
  }
)
