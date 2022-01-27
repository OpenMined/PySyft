import { useMemo } from 'react'
import type { Ref, MutableRefObject } from 'react'

type ReactRef<T> = Ref<T> | MutableRefObject<T>

function assignValueToRef<T = any>(ref: ReactRef<T> | undefined, value: T) {
  if (typeof ref === 'function') {
    ref(value)
    return
  }
  try {
    // @ts-ignore
    ref.current = value
  } catch (error) {
    throw new Error(`Cannot assign value "${value}" to ref "${ref}"`)
  }
}

/**
 * useMergeRefs
 * Custom Hook that merges React refs into a single memoized value
 *
 * We propose to use it when the component exposes a forward ref
 * and you need to use a ref for the same Element
 *
 * @example
 * const Component = forwardRef((props, ref) => {
 *   const innerRef = useRef();
 *   return <div {...props} ref={useMergeRefs(innerRef, ref)} />;
 * });
 */
export function useMergeRefs<T>(...refs: (ReactRef<T> | undefined)[]) {
  return useMemo(() => {
    if (refs.every(ref => ref == null)) return null
    return (node: T) => {
      refs.forEach(ref => {
        if (ref) assignValueToRef(ref, node)
      })
    }
  }, [refs])
}
