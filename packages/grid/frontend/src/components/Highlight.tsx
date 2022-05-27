import cn from 'classnames'
import type { ReactNode } from 'react'

export function Highlight({
  children,
  className,
}: {
  className?: string
  children: ReactNode
}) {
  return (
    <span
      className={cn(
        'bg-gray-100 text-trueGray-800 text-xs tracker-tighter p-1',
        className
      )}
    >
      {children}
    </span>
  )
}
