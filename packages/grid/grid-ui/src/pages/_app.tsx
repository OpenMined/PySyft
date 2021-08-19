import {ReactQueryDevtools} from 'react-query/devtools'
import {AppProviders} from '@/context'
import {CheckAuthRoute} from '@/components/auth-route'
import type {AppProps} from 'next/app'

import '@/styles/globals.css'

function getVersion(): string {
  return process.env.VERSION
}

function getVersionHash(): string {
  return process.env.VERSION_HASH
}

console.log("Version: ", getVersion(), getVersionHash())

export default function PyGridAdmin({Component, pageProps}: AppProps) {
  return (
    <AppProviders>
      <CheckAuthRoute>
        <Component {...pageProps} />
      </CheckAuthRoute>
      {process.env.ENVIRONMENT === 'development' && <ReactQueryDevtools initialIsOpen={false} />}
    </AppProviders>
  )
}
