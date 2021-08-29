import type {FunctionComponent} from 'react'
import {QueryCache, QueryClient, QueryClientProvider} from 'react-query'
import {AuthProvider} from '@/context/auth-context'
import api from '@/utils/api-axios'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      queryFn: async ({queryKey}) => {
        const {data} = await api(queryKey[0])
        return data
      },
      refetchInterval: 5 * 60 * 1000
    }
  },
  queryCache: new QueryCache()
})

const AppProviders: FunctionComponent = ({children}) => (
  <QueryClientProvider client={queryClient}>
    <AuthProvider>{children}</AuthProvider>
  </QueryClientProvider>
)

export {AppProviders}
