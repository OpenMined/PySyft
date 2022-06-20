import { useState, useEffect, useRef, ReactNode } from 'react'
import { createPortal } from 'react-dom'
import cn from 'classnames'
import { Overlay } from '@/components/Overlay'
import { Text } from '@/omui'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  faCheckCircle,
  faExclamationTriangle,
  faExpandAlt,
  faTimes,
} from '@fortawesome/free-solid-svg-icons'

function ModalRoot({ children }) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    document.querySelector('html').style.overflowY = 'hidden'

    return () => {
      setMounted(false)
      document.querySelector('html').style.overflowY = 'scroll'
    }
  }, [])

  return mounted
    ? createPortal(children, document.getElementById('omui-modal-portal'))
    : null
}

function useOutsideClick(ref, onClose) {
  useEffect(() => {
    function handleClickOutside(event) {
      if (ref.current && !ref.current.contains(event.target)) {
        onClose()
      }
    }

    document.addEventListener('mousedown', handleClickOutside)

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [ref])
}

function CloseModalButton({ onClose }) {
  return (
    <div className="w-full text-right">
      <button
        onClick={onClose}
        className="w-6 h-6 text-center"
        aria-label="Close"
      >
        <FontAwesomeIcon
          icon={faTimes}
          title="Close"
          className="cursor-pointer text-sm text-gray-500"
        />
      </button>
    </div>
  )
}

export default function Modal({
  className = '',
  children,
  withScrim = false,
  withExpand = null,
  show = false,
  onClose = () => {},
}: ModalProps) {
  const wrapper = useRef(null)
  useOutsideClick(wrapper, onClose)

  if (!show) return null

  return (
    <ModalRoot>
      <Overlay>
        <div className="flex items-center justify-center py-10 h-full relative">
          <div
            ref={wrapper}
            className={cn(
              'z-50 overflow-auto max-h-full cursor-auto w-full',
              className
            )}
            style={{ marginLeft: 270 }}
          >
            <div
              className={cn(
                'grid grid-cols-12 px-6 py-4 shadow-modal rounded mx-auto sm:max-w-modal lg:max-w-mbig',
                withScrim ? 'bg-scrim-white' : 'bg-white'
              )}
            >
              <div className="col-span-full flex space-x-3 justify-between">
                {withExpand && (
                  <Text bold size="sm" className="text-gray-400 w-full">
                    <FontAwesomeIcon icon={faExpandAlt} className="mr-3" />
                    Expand Page
                  </Text>
                )}
                <CloseModalButton onClose={onClose} />
              </div>
              {children}
            </div>
          </div>
        </div>
      </Overlay>
    </ModalRoot>
  )
}

interface ModalProps {
  children: ReactNode
  className?: string
  onClose: () => void
  show: boolean
  withScrim?: boolean
  withExpand?: any
}

function ModalWarning({ children, ...props }: ModalProps) {
  return (
    <Modal {...props}>
      <div className="col-span-full text-center">
        <FontAwesomeIcon
          icon={faExclamationTriangle}
          className="text-warning-500 text-3xl"
        />
      </div>
      {children}
    </Modal>
  )
}

function ModalSuccess({ children, ...props }: ModalProps) {
  return (
    <Modal {...props}>
      <div className="col-span-full text-center">
        <FontAwesomeIcon
          icon={faCheckCircle}
          className="text-success-500 text-3xl"
        />
      </div>
      {children}
    </Modal>
  )
}

function ModalGrid({ children, ...props }: ModalProps) {
  return (
    <Modal {...props}>
      <div className="grid grid-cols-12 w-full">
        <CloseModalButton onClose={props.onClose} />
        {children}
      </div>
    </Modal>
  )
}

export const Modals = Object.assign(Modal, {
  Warning: ModalWarning,
  Success: ModalSuccess,
  Grid: ModalGrid,
})
