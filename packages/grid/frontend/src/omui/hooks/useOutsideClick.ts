import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'

export type UseOutsideClickProps = {
  ref: RefObject<HTMLElement>
  callback?: (e: Event) => void
}

/**
 * useOutsideClick
 * Custom Hook that receive a Ref and a callback
 *
 * The callback is invoked when a click occurs outside the ref Element
 *
 * @example
 * const Component = (props) => {
 *   const ref = useRef()
 *   useOutsideClick({ ref, callback: () => console.log('Clicked outside ref') })
 *   return <div {...props} ref={useMergeRefs(innerRef, ref)} />;
 * });
 */
export function useOutsideClick({ ref, callback }: UseOutsideClickProps) {
  const stateRef = useRef({
    isPointerDown: false,
    ignoreEmulatedMouseEvents: false,
  })

  const state = stateRef.current

  useEffect(() => {
    const onPointerDown: any = (e: PointerEvent) => {
      if (isValidEvent(e, ref)) {
        state.isPointerDown = true
      }
    }

    const onMouseUp: any = (event: MouseEvent) => {
      if (state.ignoreEmulatedMouseEvents) {
        state.ignoreEmulatedMouseEvents = false
        return
      }

      if (state.isPointerDown && callback && isValidEvent(event, ref)) {
        state.isPointerDown = false
        callback(event)
      }
    }

    const onTouchEnd = (event: TouchEvent) => {
      state.ignoreEmulatedMouseEvents = true
      if (callback && state.isPointerDown && isValidEvent(event, ref)) {
        state.isPointerDown = false
        callback(event)
      }
    }

    document.addEventListener('mousedown', onPointerDown, true)
    document.addEventListener('mouseup', onMouseUp, true)
    document.addEventListener('touchstart', onPointerDown, true)
    document.addEventListener('touchend', onTouchEnd, true)

    return () => {
      document.removeEventListener('mousedown', onPointerDown, true)
      document.removeEventListener('mouseup', onMouseUp, true)
      document.removeEventListener('touchstart', onPointerDown, true)
      document.removeEventListener('touchend', onTouchEnd, true)
    }
  }, [callback, ref, state])
}

function isValidEvent(event: any, ref: RefObject<HTMLElement>) {
  if (event.button > 0) {
    return false
  }

  if (event.target) {
    const ownerDocument = event.target.ownerDocument
    if (
      !ownerDocument ||
      !ownerDocument.documentElement.contains(event.target)
    ) {
      return false
    }
  }

  return !ref.current?.contains(event.target)
}
