import React from 'react'
import cn from 'classnames'
import type { PropsWithRef } from 'react'

import { Divider } from '../Divider/Divider'
import { H5 } from '../Typography/Text'

type Props = {
  header?: string
  containerProps?: JSX.IntrinsicElements['div']
  containerClasses?: string
} & JSX.IntrinsicElements['aside']

export type SidebarProps = PropsWithRef<Props>

function Sidebar({
  header,
  className,
  children,
  containerProps,
  containerClasses,
  ...props
}: SidebarProps) {
  return (
    <div
      {...containerProps}
      className={cn('omui-sidebar w-full flex', header ? 'flex-col' : 'flex-row', containerClasses)}
    >
      {header && (
        <>
          <H5 className="text-gray-800 dark:text-white">{header}</H5>
          <Divider className="my-2" />
        </>
      )}
      <aside className={cn('flex flex-col flex-1', !header && 'py-6', className)} {...props}>
        {children}
      </aside>
    </div>
  )
}

export { Sidebar }
