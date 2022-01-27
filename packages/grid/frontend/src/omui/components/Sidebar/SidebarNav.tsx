import React from 'react'
import cn from 'classnames'
import type { PropsWithRef } from 'react'

import { Divider } from '../Divider/Divider'
import { H5 } from '../Typography/Text'

type Props = {
  header?: string
  containerProps?: JSX.IntrinsicElements['div']
} & JSX.IntrinsicElements['aside']

export type SidebarProps = PropsWithRef<Props>

function Sidebar({ className, children, containerProps, ...props }: SidebarProps) {
  return (
    <div className="omui-sidebar dark:bg-gray-800 w-full flex flex-col" {...containerProps}>
      {header ? (
        <>
          <H5 className="text-gray-800 dark:text-white">{header}</H5>
          <Divider className="my-2" />
        </>
      ) : (
        <Divider color="light" orientation="vertical" className="mr-6 dark:border-gray-700" />
      )}
      <aside className={cn('flex-1', !header && 'py-6', className)} {...props}>
        {children}
      </aside>
    </div>
  )
}

export { Sidebar }
