import cn from 'classnames'
import type { PropsWithChildren } from 'react'

export interface BadgeProps {
  bgColor?: string // TODO: color types
  textColor?: string
  className?: string | string[]
}

export function Badge({
  bgColor = 'gray',
  textColor = 'gray',
  className,
  children,
}: PropsWithChildren<BadgeProps>) {
  return (
    <div
      className={cn(
        `bg-${bgColor}-500`,
        `text-${textColor}-50`,
        'uppercase text-xs tracking-tighter rounded-lg px-2 py-1 leading-3 h-auto w-auto text-center font-semibold my-auto',
        className
      )}
    >
      <span className="inline-block">{children}</span>
    </div>
  )
}
