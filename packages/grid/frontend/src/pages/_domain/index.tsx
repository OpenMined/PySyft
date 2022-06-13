import { useRouter } from 'next/router'
import { useSettings } from '@/lib/data'
import { getToken } from '@/lib/auth'

export default function Home() {
  const router = useRouter()
  const isAuthenticated = getToken()
  const { data: settings } = useSettings().all()

  if (typeof window !== 'undefined' && settings) {
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
