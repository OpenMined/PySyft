import cn from 'classnames'
import type { ReactNode } from 'react'

export interface Tag {
  className?: string
  children: ReactNode
}

export function Tag({ className, children }: Tag) {
  return (
    <div
      className={cn(
        'inline-block px-2 py-1 text-sm border border-gray-200 shadow-sm text-gray-800 rounded-md flex-shrink-0',
        className,
        'bg-blueGray-100'
      )}
    >
      {children}
    </div>
  )
}
