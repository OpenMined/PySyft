import { useCallback } from 'react'
import 'twin.macro'
import Link from 'next/link'
import { useForm } from 'react-hook-form'
import { define, string, object } from 'superstruct'
import toast from 'react-hot-toast'
import { Button } from '$components/Buttons'
import { FormControl } from '$components/FormControl'
import { DomainStatus } from '$components/DomainStatus'
import { AuthLayout } from '$components/AuthLayout'
import { api } from '$lib/api'
import isEmail from 'is-email'

import type { RHFElements } from '$types'

// https://github.com/react-hook-form/resolvers/issues/271
const { superstructResolver } = require('@hookform/resolvers/superstruct')

interface LoginFormFields {
  email: string
  password: string
}

const Login = () => {
  return (
    <AuthLayout>
      <div tw="shadow-card-neutral-1 bg-layout-white bg-scrim-layout-white px-2.5 pt-2 pb-4">
        <div tw="p-6 text-center">
          <span tw="text-2xl font-rubik">Welcome Back</span>
          <div data-cy="domain-status" tw="text-sm">
            <DomainStatus status="online" />
          </div>
        </div>
        <LoginForm />
      </div>
    </AuthLayout>
  )
}

export default Login

const Email = define('email', isEmail)

const schema = object({
  email: Email,
  password: string(),
})

const LoginForm = () => {
  const {
    register,
    handleSubmit,
    formState: { errors, isValid, isSubmitting },
  } = useForm<LoginFormFields>({
    mode: 'onBlur',
    reValidateMode: 'onBlur',
    resolver: superstructResolver(schema),
  })
  const isFormValid = isValid && !isSubmitting

  const login = useCallback(async (data: any) => {
    // @ts-ignore
    const loginToast = toast('Logging in...', {
      toastType: 'info',
      variant: 'dark',
      position: 'top-left',
    })
    try {
      const res = await api.post('login', {
        json: data,
      })
      return res.json() // ok
    } catch (e) {
      toast('Invalid credentials', {
        // @ts-ignore
        toastType: 'danger',
        variant: 'accent',
        id: loginToast,
        message: 'Invalid credentials',
        title: 'Login error',
        position: 'top-left',
      })
    }
  }, [])

  return (
    <form onSubmit={handleSubmit(login)}>
      <div tw="flex flex-col gap-4 px-8">
        {loginForm.map((item: RHFElements) => (
          <FormControl
            key={item.name}
            {...item}
            {...register(item.name as keyof LoginFormFields, {
              required: item.required,
              ...item.registerOptions,
            })}
            error={errors?.[item.name]?.message}
          />
        ))}
        <p data-cy="redirect-sign-up" tw="text-center mt-2">
          Don't have an account yet?{' '}
          <Link href="/register">
            <a>Apply for an account here.</a>
          </Link>
        </p>
      </div>
      <div tw="flex justify-center p-6 pb-4">
        <Button data-cy="login-button" disabled={!isFormValid}>
          Login
        </Button>
      </div>
    </form>
  )
}

const loginForm: Array<RHFElements> = [
  {
    name: 'email',
    label: 'Email',
    type: 'text',
    placeholder: 'abc@university.org',
    autoComplete: 'email',
    required: true,
  },
  {
    name: 'password',
    label: 'Password',
    type: 'password',
    placeholder: '*********',
    autoComplete: 'current-password',
    required: true,
  },
]
