import {useState} from 'react'
import {QueryClient, QueryClientProvider} from 'react-query'
import {AuthProvider} from '@/context/auth-context'

export default function AppProviders({children}) {
  const [queryClient] = useState(() => new QueryClient())

  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>{children}</AuthProvider>
    </QueryClientProvider>
  )
}
