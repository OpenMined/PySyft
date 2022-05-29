import { createContext, useContext, useState } from 'react'
import cn from 'classnames'
import Link from 'next/link'
import { useForm } from 'react-hook-form'
import { CardItem } from '@/components/CardItem'
import { Badge, Button, Divider, Input, H1, Text } from '@/omui'
import { Footer, Tags } from '@/components/lib'

import type { Domain } from '@/types/domain'

import loginStrings from 'i18n/en/signup.json'
import signUpStrings from 'i18n/en/signup.json'
import commonStrings from 'i18n/en/common.json'

import { useSettings, useUsers } from '@/lib/data'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faDownload, faPlus } from '@fortawesome/free-solid-svg-icons'
import { FormControl } from '@/omui/components/FormControl/FormControl'

import ky from 'ky'

const SignUpContext = createContext<{
  domain: Domain
  error: string
  loading: boolean
  signup: ({
    email,
    password,
  }: {
    email: string
    password: string
  }) => Promise<any>
}>({
  domain: null,
  signup: null,
  error: null,
  loading: null,
})

function Container({ children }) {
  return (
    <div className="flex items-center justify-center py-10 h-full relative">
      <div className={cn('z-50 overflow-auto max-h-full cursor-auto w-full')}>
        <div
          className={cn(
            'grid grid-cols-12 px-6 py-4 shadow-modal rounded mx-auto sm:max-w-modal lg:max-w-mbig',
            'bg-white'
          )}
        >
          {children}
        </div>
      </div>
    </div>
  )
}

function SignUpBox({ withDAA = false }) {
  const { domain } = useContext(SignUpContext)

  const {
    handleSubmit,
    register,
    formState: { isValid, isDirty },
  } = useForm({ mode: 'onChange' })

  const isDisabled = !isValid || !isDirty
  const [file, setFile] = useState(null)

  const signup = (values) => {
    const formData = new FormData()
    formData.append('new_user', JSON.stringify({ ...values, role: 1 }))
    // TODO: unauth api connect
    ky.post('/api/v1/register', { body: formData })
  }

  return (
    <section className="flex flex-col items-center space-y-4">
      <Text
        size={withDAA ? '2xl' : '5xl'}
        className={cn(withDAA && 'w-full text-left')}
      >
        {withDAA ? signUpStrings['apply-for-account'] : domain?.domain_name}
      </Text>
      <fieldset className="w-full">
        <form onSubmit={handleSubmit(signup)}>
          <div className="grid grid-cols-2 gap-x-6 gap-y-4">
            <FormControl id="name" label={commonStrings['full-name']} required>
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
              <FormControl label={commonStrings.email} required>
                <Input
                  placeholder={commonStrings.placeholder.email}
                  {...register('email', { required: true })}
                />
              </FormControl>
            </div>
            <FormControl label={commonStrings.password} required>
              <Input
                type="password"
                placeholder="···········"
                {...register('password', { required: true })}
              />
            </FormControl>
            <FormControl
              label={commonStrings['confirm-password']}
              id="confirm_password"
              required
            >
              <Input
                type="password"
                placeholder="···········"
                {...register('confirm_password', { required: true })}
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
            {withDAA && (
              <div className="col-span-full space-y-4">
                <Text bold className="text-gray-500">
                  {signUpStrings['with-daa']['upload-signed']}
                  <Text className="text-primary-600">*</Text>
                </Text>
                <Text as="p">{signUpStrings['with-daa'].description}</Text>
                <div>
                  <Button variant="outline" size="sm">
                    <Text size="sm" bold>
                      <FontAwesomeIcon icon={faPlus} className="mr-2" />
                      {file
                        ? commonStrings.buttons['replace-file']
                        : commonStrings.buttons['upload-file']}
                    </Text>
                  </Button>
                  <Button variant="link">
                    <Text size="sm" bold>
                      <FontAwesomeIcon icon={faDownload} className="mr-2" />
                      {commonStrings.buttons['download-agreement']}
                    </Text>
                  </Button>
                </div>
              </div>
            )}
          </div>
          <div className="space-y-4 text-center mt-10">
            <Divider color="light" />
            <Button
              type="submit"
              size="sm"
              className="my-6 mb-4"
              disabled={isDisabled}
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
    </section>
  )
}

function DomainInfoHeader() {
  const { domain } = useContext(SignUpContext)
  return (
    <>
      <Tags tags={domain?.tags} />
      <H1 className="mt-2">{domain?.name}</H1>
    </>
  )
}

function DomainInfoDetails() {
  const { domain } = useContext(SignUpContext)
  const information = [
    {
      text: 'ID#',
      value: domain.id,
      ValueComponent: (props) => (
        <Badge variant="gray" type="subtle" {...props} />
      ),
    },
    {
      text: commonStrings['hosted-datasets'],
      value: domain?.total_datasets,
      ValueComponent: (props) => <Text mono {...props} />,
    },
    {
      text: commonStrings['deployed-on'],
      value: domain?.created_on,
      ValueComponent: (props) => <Text mono {...props} />,
    },
    {
      text: commonStrings['owner'],
      value: `${domain?.owner}, ${domain?.company}`,
      ValueComponent: (props) => <Text mono {...props} />,
    },
  ]
  return (
    <section className="mt-10 space-y-4">
      {information.map((info) => (
        <CardItem
          key={info.text}
          text={info.text}
          value={info.value}
          ValueComponent={info.ValueComponent}
        />
      ))}
    </section>
  )
}

function DomainInfoSupport() {
  const { domain } = useContext(SignUpContext)

  if (!domain?.email) return null

  return (
    <div className="mt-8">
      <Divider color="light" />
      <Text as="p" size="sm">
        {loginStrings['support-email']}
      </Text>
      <a href={`mailto:${domain.email}`}>
        <Text
          as="p"
          size="sm"
          className="text-primary-600 hover:text-primary-500"
        >
          {domain?.email}
        </Text>
      </a>
    </div>
  )
}

function DomainInfo() {
  const { domain } = useContext(SignUpContext)
  return (
    <>
      <header className="col-span-4 col-start-2 mb-6">
        <DomainInfoHeader />
      </header>
      <div className="col-span-4 col-start-2">
        <Text as="p">{domain?.description}</Text>
        <DomainInfoDetails />
        <DomainInfoSupport />
      </div>
    </>
  )
}

function TopRightArt() {
  return (
    <div className="absolute w-screen h-screen inset-0">
      <div className="relative w-full h-full">
        <div
          className="absolute overflow-hidden z-10"
          style={{
            right: -92,
            top: -13,
            width: 246,
            height: 246,
            background:
              'linear-gradient(90deg, rgba(255, 255, 255, 0.5) 0%, rgba(255, 255, 255, 0) 100%), #20AFDF',
            filter: 'blur(70px)',
          }}
        />
        <div
          className="absolute overflow-hidden z-20"
          style={{
            width: 730,
            height: 780,
            top: 36,
            right: -92,
            background:
              'linear-gradient(90deg, rgba(255, 255, 255, 0.5) 0%, rgba(255, 255, 255, 0) 100%), #EB4913',
            filter: 'blur(70px)',
          }}
        />
        <div
          className="absolute overflow-hidden z-30"
          style={{
            width: 808,
            height: 808,
            top: 61,
            right: -92,
            background:
              'linear-gradient(90deg, rgba(255, 255, 255, 0.5) 0%, rgba(255, 255, 255, 0) 100%), #EC9913',
            filter: 'blur(180px)',
          }}
        />
      </div>
    </div>
  )
}

function SignUpPageWithDAA() {
  return (
    <article
      className="grid px-7 grid-cols-12 grid-rows-3 min-h-screen w-full overflow-hidden"
      style={{
        gridTemplateAreas: `
          "header"
          "content"
          "footer"
        `,
        gridTemplateRows: '140px auto 80px',
      }}
    >
      <div className="col-span-full col-start-2 mt-10">
        {/* Sub with layout above ^^ */}
        <img src="/assets/small-logo.png" width={100} />
        <TopRightArt />
      </div>
      <div className="grid grid-cols-12 col-span-full z-40 content-start">
        <DomainInfo />
        <div
          className="col-span-5 col-end-12 shadow-modal p-8"
          style={{
            background:
              'linear-gradient(90deg, rgba(255, 255, 255, 0.8) 0%, rgba(255, 255, 255, 0.5) 100%), #F1F0F4',
          }}
        >
          <SignUpBox withDAA />
        </div>
      </div>
      <Footer className="col-start-2 col-span-full" />
    </article>
  )
}

function SignUpPageWithoutDAA() {
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
        <SignUpBox />
      </div>
      <Footer className="col-span-full justify-end" />
    </article>
  )
}

export default function SignUp() {
  const { data: domain } = useSettings().all()
  const [error, setError] = useState(null)

  const handleSignUp = async (values) => {
    try {
      await signup(values)
    } catch (err) {
      setError(err)
    }
  }

  return (
    <SignUpContext.Provider
      value={{ domain, error, loading: null, signup: handleSignUp }}
    >
      {domain?.daa ? <SignUpPageWithDAA /> : <SignUpPageWithoutDAA />}
    </SignUpContext.Provider>
  )
}
