import React, { useState } from 'react'
import { ReactQueryDevtools } from 'react-query/devtools'
import { Hydrate, QueryClient, QueryClientProvider } from 'react-query'
import { config } from '@fortawesome/fontawesome-svg-core'
import type { AppProps } from 'next/app'

import '@/styles/globals.css'
import '../../node_modules/@fortawesome/fontawesome-svg-core/styles.css'

config.autoAddCss = false

export default function PyGridUI({ Component, pageProps }: AppProps) {
  const [queryClient] = useState(() => new QueryClient())
  return (
    <QueryClientProvider client={queryClient}>
      <Hydrate state={pageProps.dehydratedState}>
        <Component {...pageProps} />
      </Hydrate>
      {process.env.NEXT_PUBLIC_ENVIRONMENT !== 'development' && (
        <ReactQueryDevtools initialIsOpen={false} />
      )}
    </QueryClientProvider>
  )
}
