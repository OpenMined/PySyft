import { useState } from 'react'
import Link from 'next/link'
import { useForm } from 'react-hook-form'
import { Button, Divider, Input, Text } from '@/omui'
import { Footer } from '@/components/lib'

import signUpStrings from 'i18n/en/signup.json'
import commonStrings from 'i18n/en/common.json'

import { useSettings } from '@/lib/data'
import { FormControl } from '@/omui/components/FormControl/FormControl'

import ky from 'ky'
import { EyeOpen, EyeShut } from '@/components/EyeIcon'

export default function SignUp() {
  const { data: domain } = useSettings().all()
  const [isLoading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const {
    handleSubmit,
    register,
    formState: { isValid, isDirty },
  } = useForm({ mode: 'onChange' })

  const [isPasswordVisible, setPasswordVisible] = useState(false)
  const isDisabled = !isValid || !isDirty

  const signup = async (values) => {
    try {
      setLoading(true)
      const formData = new FormData()
      formData.append('new_user', JSON.stringify({ ...values, role: 1 }))
      // TODO: unauth api connect
      await ky.post('/api/v1/register', { body: formData })
    } catch (e) {
      setError(e?.message ?? 'Error registering new user')
    } finally {
      setLoading(false)
    }
  }

  return (
    <article
      className="grid px-7 grid-cols-12 grid-rows-3 min-h-screen w-full"
      style={{
        gridTemplateAreas: `
          "header"
          "content"
          "footer"
        `,
        gridTemplateRows: 'minmax(min-content, 200px) auto 80px',
      }}
    >
      <div className="col-span-full self-end justify-self-center">
        <img src="/assets/small-grid-symbol-logo.png" width={80} height={80} />
      </div>
      <div className="col-span-6 col-start-4 mt-8">
        <section className="flex flex-col items-center space-y-4">
          <Text size="5xl">{domain?.domain_name}</Text>
          <fieldset className="w-full">
            <form onSubmit={handleSubmit(signup)}>
              <div className="grid grid-cols-2 gap-x-6 gap-y-4">
                <FormControl
                  id="name"
                  label={commonStrings['full-name']}
                  required
                >
                  <Input
                    placeholder={commonStrings.placeholder['full-name']}
                    {...register('name', { required: true })}
                  />
                </FormControl>
                <FormControl
                  id="institution"
                  label={commonStrings['company-institution']}
                  optional
                >
                  <Input
                    placeholder="ABC University"
                    {...register('institution')}
                  />
                </FormControl>
                <div className="col-span-full">
                  <FormControl id="email" label={commonStrings.email} required>
                    <Input
                      placeholder={commonStrings.placeholder.email}
                      {...register('email', { required: true })}
                    />
                  </FormControl>
                </div>
                <FormControl
                  id="password"
                  label={commonStrings.password}
                  required
                >
                  <Input
                    type={isPasswordVisible ? 'text' : 'password'}
                    placeholder={isPasswordVisible ? 'password' : '···········'}
                    {...register('password', { required: true })}
                    addonRight={
                      <button
                        type="button"
                        onClick={() => setPasswordVisible(!isPasswordVisible)}
                      >
                        {isPasswordVisible ? <EyeOpen /> : <EyeShut />}
                      </button>
                    }
                  />
                </FormControl>
                <FormControl
                  label={commonStrings['confirm-password']}
                  id="confirm_password"
                  required
                >
                  <Input
                    type={isPasswordVisible ? 'text' : 'password'}
                    placeholder={isPasswordVisible ? 'password' : '···········'}
                    {...register('confirm_password', { required: true })}
                    addonRight={
                      <button
                        type="button"
                        onClick={() => setPasswordVisible(!isPasswordVisible)}
                      >
                        {isPasswordVisible ? <EyeOpen /> : <EyeShut />}
                      </button>
                    }
                  />
                </FormControl>
                <div className="col-span-full">
                  <FormControl
                    label={commonStrings['website-profile']}
                    id="website"
                    optional
                  >
                    <Input
                      placeholder={commonStrings['website-profile']}
                      {...register('website')}
                    />
                  </FormControl>
                </div>
              </div>
              <div className="space-y-4 text-center mt-10">
                <Divider color="light" />
                <Button
                  type="submit"
                  size="sm"
                  className="my-6 mb-4"
                  disabled={isDisabled}
                  isLoading={isLoading}
                >
                  {commonStrings.buttons['submit-application']}
                </Button>
                <div>
                  <Text size="sm">{signUpStrings['have-an-account']}</Text>{' '}
                  <Link href="/login">
                    <a>
                      <Text size="sm" underline className="text-primary-600">
                        {signUpStrings['login-here']}
                      </Text>
                    </a>
                  </Link>
                </div>
              </div>
            </form>
          </fieldset>
          {error && <div className="text-error-600">{error}</div>}
        </section>
      </div>
      <Footer className="col-span-full justify-end" />
    </article>
  )
}
