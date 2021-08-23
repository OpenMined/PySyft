import {Sidebar} from '@/components'
import type {PropsWithChildren} from 'react'

interface PageProps {
  title?: string
  description?: string
}

export function Page({title, description, children}: PropsWithChildren<PageProps>) {
  return (
    <article className="md:flex max-w-7xl justify-self-center mx-auto">
      <Sidebar />
      <main className="w-full h-full p-4 px-4 sm:px-6 md:px-8 space-y-6 lg:space-y-8">
        {title && (
          <header className="h-16 mt-2">
            <h1 className="text-3xl">{title}</h1>
            <p className="text-gray-500">{description}</p>
          </header>
        )}
        {children}
      </main>
    </article>
  )
}
