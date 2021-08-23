import {useRouter} from 'next/router'
import {useAuth} from '@/context/auth-context'
import {useDomainStatus} from '@/lib/data'

export default function Home() {
  const router = useRouter()
  const {getToken} = useAuth()
  const isAuthenticated = getToken()
  const {data} = useDomainStatus()

  if (typeof window !== 'undefined' && data) {
    if (isAuthenticated) {
      router.replace('/users')
      return null
    } else {
      router.replace('/login')
      return null
    }
  }

  return null
}
