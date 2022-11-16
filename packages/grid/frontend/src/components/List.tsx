import Link from 'next/link'
import cn from 'classnames'
import type { ReactNode } from 'react'

interface ListItem {
  href?: string
  className?: string
  children: ReactNode
}

function ListRoot({ children }: { children: ReactNode }) {
  return (
    <ul className="grid grid-cols-1 divide-y divide-gray-200 border border-gray-200">
      {children}
    </ul>
  )
}

function ListItem({ href, className, children }: ListItem) {
  if (href) {
    return (
      <ListItemWithLink href={href} className={className}>
        {children}
      </ListItemWithLink>
    )
  }
  return (
    <li className={cn('px-4 py-5 bg-white hover:bg-sky-100', className)}>
      {children}
    </li>
  )
}

export function ListItemWithLink({ href, className, children }: ListItem) {
  return (
    <Link href={href}>
      <a>
        <ListItem className={cn('cursor-pointer', className)}>
          {children}
        </ListItem>
      </a>
    </Link>
  )
}

export const List = Object.assign(ListRoot, { Item: ListItem })
