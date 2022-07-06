import { useEffect } from 'react'
import { useRouter } from 'next/router'
import { LoadingPyGrid } from '@/components'
import { useAuth } from '@/context/auth-context'
import { useDomainStatus } from '@/lib/data'
import type { ReactNode } from 'react'

interface Pages {
  children: ReactNode
}

export function CheckAuthRoute({ children }: Pages) {
  const router = useRouter()
  const { getToken } = useAuth()
  const { data, isError } = useDomainStatus()
  const publicRoutes = ['/status', '/offline', '/login', '/start']
  const isPublicRoute = publicRoutes.includes(router.route)

  console.log({ data })

  useEffect(() => {
    // if (data) {
    //   router.push('/start')
    //   return null
    // }

    if (!isPublicRoute) {
      if (isError) {
        router.push('/offline')
        return null
      }

      const token = getToken()

      if (!token) {
        router.push('/login')
        return null
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data])

  if (!data && !isPublicRoute) {
    return <LoadingPyGrid />
  }

  return <>{children}</>
}
