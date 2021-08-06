import {useState, useRef} from 'react'
import {useRouter} from 'next/router'
import {useForm} from 'react-hook-form'
import {DomainConnectionStatus} from '@/components'
import {useDomainStatus} from '@/lib/data'
import {useAuth} from '@/context/auth-context'
import {Input, NormalButton} from '@/components'

interface UserLogin {
  email: string
  password: string
}

export default function Login() {
  const router = useRouter()
  const {login} = useAuth()
  const {register, handleSubmit, formState} = useForm({mode: 'onChange'})
  const [error, setError] = useState<boolean>(false)
  const [loading, setLoading] = useState<boolean>(false)
  const [spin, setSpin] = useState<boolean>(false)
  const rotateStyle = useRef({})
  const {isValid} = formState
  const {data: status} = useDomainStatus()

  if (spin) {
    rotateStyle.current = {transform: `rotate(${Math.ceil(365 * Math.random())}deg)`}
    setSpin(false)
    if (error) {
      setError(false)
    }
  }

  const onSubmit = async (values: UserLogin) => {
    try {
      setLoading(true)
      await login(values)
      router.push('/users')
    } catch ({message}) {
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="container mx-auto">
      <div className="flex flex-col items-center max-w-md mx-auto space-y-4 text-center">
        <div className="transition transform" style={rotateStyle.current}>
          <img alt="PyGrid logo" src="/assets/logo.png" width={200} height={200} />
        </div>
        <h1>PyGrid UI</h1>
        {status && (
          <p className="text-gray-600">
            Login to <b>{status.nodeName}</b> Domain
          </p>
        )}
        <form className="w-4/5" onSubmit={handleSubmit(onSubmit)}>
          <div className="flex flex-col space-y-6 text-left">
            <div className="flex flex-col">
              <Input
                name="email"
                id="email"
                label="Email"
                ref={register({required: "Don't forget your email"})}
                placeholder="owner@openmined.org"
                error={formState.errors.email?.message}
                onChange={() => setSpin(true)}
              />
              {error && <span className="px-4 py-1 mt-0.5 text-sm text-gray-800 bg-red-200">{error}</span>}
            </div>
            <div className="flex flex-col">
              <Input
                type="password"
                id="password"
                name="password"
                label="Password"
                ref={register({required: 'This needs a password...'})}
                placeholder="••••••••••"
                onChange={() => setSpin(true)}
              />
            </div>
            <NormalButton
              className="bg-sky-500 hover:bg-sky-300 active:bg-sky-800 text-white"
              disabled={!isValid || error}
              isLoading={loading}>
              Login
            </NormalButton>
          </div>
        </form>
        <DomainConnectionStatus />
      </div>
    </main>
  )
}
