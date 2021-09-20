import {useRouter} from 'next/router'
import {useDomainStatus} from '@/lib/data'

export default function Offline() {
  const router = useRouter()
  const {data} = useDomainStatus()

  if (data) {
    router.push('/')
    return null
  }

  return (
    <main className="flex flex-col items-center justify-center h-screen px-4 py-8 mx-auto max-w-7xl">
      <h1 className="text-xl tracking-tight">The Domain is offline.</h1>
      <p>Trying to reach the API at {process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost'}.</p>
      <div>
        <img className="absolute w-20 animate-ping" alt="PyGrid logo" src="/assets/logo.png" />
        <img className="w-20" alt="PyGrid logo" src="/assets/logo.png" />
      </div>
    </main>
  )
}
