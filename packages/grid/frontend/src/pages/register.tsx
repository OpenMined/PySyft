import { useCallback } from 'react'
import tw from 'twin.macro'
import Link from 'next/link'
import toast from 'react-hot-toast'
import { useForm } from 'react-hook-form'
import { Button } from '$components/Buttons'
import { FormControl } from '$components/FormControl'
import { AuthLayout } from '$components/AuthLayout'
import { api } from '$lib/api'
import { useRouter } from 'next/router'

const registerForm = [
  {
    name: 'full_name',
    required: true,
    type: 'text',
    placeholder: 'Jane Doe',
    label: 'Full Name',
  },
  {
    name: 'company',
    optional: true,
    type: 'text',
    placeholder: 'ABC University',
    label: 'Company/Institution',
  },
  {
    name: 'email',
    label: 'Email',
    type: 'text',
    placeholder: 'abc@university.org',
    required: true,
    fullWidth: true,
  },
  {
    name: 'password',
    label: 'Password',
    type: 'password',
    placeholder: '********',
    required: true,
  },
  {
    name: 'password_confirmation',
    label: 'Confirm Password',
    type: 'password',
    placeholder: '********',
    required: true,
  },
  {
    name: 'website',
    label: 'Website/Profile',
    type: 'text',
    placeholder: 'This can help a domain owner vett your application',
    optional: true,
    fullWidth: true,
  },
]

const Register = () => {
  const { router } = useRouter()
  const { register, handleSubmit } = useForm({ mode: 'onChange' })

  const signup = useCallback(async data => {
    const { email, password, full_name, company, website } = data
    try {
      const res = await api.post('register', {
        json: { email, password, full_name, company, website },
      })
      router.push('/login')
    } catch (err) {
      toast('User already registered', {
        toastType: 'danger',
        variant: 'accent',
        id: 'signup-error',
        title: 'Sign up error',
        position: 'top-left',
      })
    }
  }, [])

  return (
    <AuthLayout>
      <div tw="shadow-card-neutral-1 bg-layout-white bg-scrim-layout-white px-2.5 pt-2 pb-4">
        <header tw="p-6">
          <span tw="text-2xl font-rubik">Apply for an Account</span>
        </header>
        <form onSubmit={handleSubmit(signup)}>
          <div tw="grid grid-cols-2 gap-x-6 gap-y-4 px-8">
            {registerForm.map(field => (
              <div key={field.name} css={[field.fullWidth && tw`col-span-2`]}>
                <FormControl {...field} />
              </div>
            ))}
          </div>
          <footer tw="flex flex-col text-center p-6 py-4">
            <Button data-cy="register-button">Submit Application</Button>
            <p tw="text-center mt-6">
              Have an account already?{' '}
              <Link href="/login">
                <a>Login here</a>
              </Link>
            </p>
          </footer>
        </form>
      </div>
    </AuthLayout>
  )
}

export default Register
