import {useRouter} from 'next/router'
import {useForm} from 'react-hook-form'
import {Input, NormalButton, DomainConnectionStatus, MutationError} from '@/components'
import {useInitialSetup} from '@/lib/data'

interface Onboarding {
  domainName: string
  email: string
  password: string
}

export default function Onboarding() {
  const router = useRouter()
  const {
    register,
    handleSubmit,
    formState: {isValid}
  } = useForm({mode: 'onTouched'})

  const create = useInitialSetup()
  const mutation = create({
    onSuccess: () => {
      router.push('/login')
      return
    }
  })

  const onSubmit = ({email, password, domainName}: Onboarding) => {
    mutation.mutate({email, password, nodeName: domainName})
  }

  return (
    <main className="flex flex-col items-center justify-center min-h-screen mx-auto max-w-7xl">
      <div className="p-12 m-8 space-y-6 rounded-lg bg-blueGray-200 md:min-w-lg">
        <header className="space-y-3">
          <h1>PyGrid</h1>
          <p className="text-gray-500">One last step to finish the Domain setup</p>
        </header>
        <p className="max-w-lg">
          After installing PyGrid, you must set up a Domain owner account and give your Domain a name.
        </p>
        <form className="max-w-lg space-y-3" onSubmit={handleSubmit(onSubmit)}>
          <MutationError
            isError={mutation.isError}
            error="Unable to setup domain"
            description={mutation.error?.message}
          />
          <Input label="Domain name" id="domainName" name="domainName" placeholder="Grid Domain" ref={register} />
          <Input label="Email or username" placeholder="owner@openmined.org" id="email" name="email" ref={register} />
          <Input
            label="Password"
            id="password"
            name="password"
            type="password"
            placeholder="••••••••••"
            ref={register}
          />
          <div className="flex items-center mt-4">
            <NormalButton
              className="w-36 disabled:bg-gray-200 disabled:text-white"
              disabled={!isValid || mutation.isLoading}
              isLoading={mutation.isLoading}>
              Start Domain
            </NormalButton>
            <DomainConnectionStatus />
          </div>
        </form>
      </div>
    </main>
  )
}
