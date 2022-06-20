import { useRouter } from 'next/router'
import Link from 'next/link'
import { useForm } from 'react-hook-form'
import { Button, Input, Text } from '@/omui'
import { DomainStatus } from '@/components/DomainStatus'
import { Footer } from '@/components/lib'
import { FormControl } from '@/omui/components/FormControl/FormControl'
import { login } from '@/lib/auth'
import { useSettings } from '@/lib/data'
import { t } from '@/i18n'
import { EyeOpen, EyeShut } from '@/components/EyeIcon'
import { useState } from 'react'
import { BetaBadge } from '@/components/BetaBadge'

export default function Login() {
  const router = useRouter()
  const [isPasswordVisible, setPasswordVisible] = useState(false)
  const [isLoading, setLoading] = useState(false)
  const { data: settings } = useSettings().all()

  const {
    handleSubmit,
    register,
    setError,
    formState: { isValid, isDirty, errors },
  } = useForm<{ email: string; password: string }>({ mode: 'onChange' })

  const handleLogin = async ({ email, password }) => {
    try {
      setLoading(true)
      await login({ email, password })
      router.push('/users')
    } catch (err) {
      setError(
        'email',
        { type: 'manual', message: 'Invalid credentials' },
        { shouldFocus: true }
      )
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
      <div className="col-span-4 col-start-5 mt-8">
        <section className="flex flex-col items-center space-y-4">
          <Text size="2xl text-center">Login to {settings?.domain_name}</Text>
          <Text className="text-gray-600 text-center">
            {t('running-version')}{' '}
            <span className="text-gray-800">
              {settings?.version ?? '0.7.0-beta.29'}
            </span>{' '}
            <BetaBadge />
          </Text>
        </section>
        <section className="mt-10 space-y-4">
          <fieldset className="w-full">
            <form onSubmit={handleSubmit(handleLogin)}>
              <FormControl
                id="email"
                label={t('email')}
                error={Boolean(errors.email)}
                hint={errors.email?.message}
                required
              >
                <Input
                  placeholder="abc@university.edu"
                  {...register('email', { required: true })}
                />
              </FormControl>
              <FormControl
                className="mt-4"
                id="password"
                label={t('password')}
                error={Boolean(errors.email)}
                required
              >
                <Input
                  type={isPasswordVisible ? 'text' : 'password'}
                  placeholder="···········"
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
              <Button
                size="sm"
                className="mt-6 w-full justify-center"
                disabled={!isValid || !isDirty}
                isLoading={isLoading}
              >
                {t('buttons.login')}
              </Button>
            </form>
          </fieldset>
        </section>
        <section className="block space-y-6 text-center mt-6">
          <DomainStatus noBox />
          <div>
            <Text size="sm">{t('no-account', 'login')} </Text>
            <Link href="/signup">
              <a className="text-link">
                <Text size="sm" underline>
                  {t('apply-account', 'login')}
                </Text>
              </a>
            </Link>
          </div>
        </section>
      </div>
      <Footer className="col-span-full justify-end" />
    </article>
  )
}
