import {ReactQueryDevtools} from 'react-query/devtools'
import {AppProviders} from '@/context'
import {CheckAuthRoute} from '@/components/auth-route'
import type {AppProps} from 'next/app'

import '@/styles/globals.css'

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
