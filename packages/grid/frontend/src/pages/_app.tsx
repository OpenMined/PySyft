import React, { useState } from 'react'
import { ReactQueryDevtools } from 'react-query/devtools'
import { Hydrate, QueryClient, QueryClientProvider } from 'react-query'
import { config } from '@fortawesome/fontawesome-svg-core'
import { Toast } from '$components/Toast'
import '$icons/FontAwesomeLibrary'

import type { AppProps } from 'next/app'

import '../styles/fonts.css'
import 'tailwindcss/tailwind.css'
import { GlobalStyles } from '../styles/GlobalStyles'
import '../../node_modules/@fortawesome/fontawesome-svg-core/styles.css'

config.autoAddCss = false

const App = ({ Component, pageProps }: AppProps) => {
  const [hydrate, setHydrate] = useState(false)

  return (
    <QueryClientProvider client={new QueryClient()}>
      <Hydrate onBeforeHydrate={() => setHydrate(true)} onAfterHydrate={() => setHydrate(false)}>
        <GlobalStyles />
        <Component {...pageProps} />
        <Toast />
        {!hydrate && <ReactQueryDevtools initialIsOpen={false} />}
      </Hydrate>
    </QueryClientProvider>
  )
}

export default App
