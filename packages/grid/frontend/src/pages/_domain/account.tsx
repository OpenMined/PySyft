import { createContext, useContext, useState } from 'react'
import { useRouter } from 'next/router'
import { useForm } from 'react-hook-form'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faCheckCircle, faExclamationTriangle } from '@fortawesome/free-solid-svg-icons'
import { Alert } from '@/components/Alert'
import { Breadcrumbs } from '@/components/Breadcrumbs'
import { Base } from '@/components/Layouts'
import { ButtonGroup, TopContent } from '@/components/lib'
import { Button, Divider, H4, Input, Text, TextArea } from '@/omui'
import { t } from '@/i18n'
import { useMe, useUsers } from '@/lib/data'
import Modal from '@/components/Modal'
import { FormControl } from '@/omui/components/FormControl/FormControl'
import { logout } from '@/lib/auth'
import { Loader } from '@/components/Loader'

import type { User } from '@/types/user'

const Heading = ({ children }) => (
  <div className="col-span-full space-y-6">
    <H4>{children}</H4>
  </div>
)

const Form = ({ isValid, children, onSubmit, reset, defaultValues = {} }) => (
  <form className="col-span-full space-y-6" onSubmit={onSubmit}>
    {children}
    <ButtonGroup>
      <Button disabled={!isValid}>{t('buttons.save-changes')}</Button>
      <Button variant="ghost" type="button" onClick={() => reset(defaultValues)}>
        {t('buttons.cancel')}
      </Button>
    </ButtonGroup>
  </form>
)

function AccountSettingsProfile() {
  const user = useAccount()
  const update = useUsers().update(user.id).mutate
  const {
    handleSubmit,
    register,
    reset,
    formState: { isValid, isDirty },
  } = useForm({
    mode: 'onChange',
    defaultValues: {
      name: user.name,
      email: user.email,
      institution: user.institution,
      website: user.website,
    },
  })

  function onSubmit(values) {
    update(values)
  }

  return (
    <section className="col-span-5 space-y-6 col-start-1">
      <Heading>{t('profile', 'account')}</Heading>
      <Form
        onSubmit={handleSubmit(onSubmit)}
        reset={reset}
        isValid={isValid && isDirty}
        defaultValues={user}
      >
        <Input label={t('full-name')} {...register('name', { required: true })} required />
        <Input label={t('email')} {...register('email', { required: true })} required />
        <Input
          label={t('company-institution')}
          {...register('institution')}
          optional
          hint={t('hints.company-institution')}
        />
        <Input
          label={t('website-profile')}
          {...register('website')}
          optional
          hint={t('hints.website-profile')}
        />
      </Form>
    </section>
  )
}

function AccountSettingsPassword() {
  const user = useAccount()
  const update = useUsers().update(user.id).mutate
  const {
    handleSubmit,
    register,
    reset,
    formState: { isValid, isDirty },
  } = useForm({ mode: 'onChange' })

  function onSubmit(values) {
    update(values)
  }

  return (
    <section className="col-span-5 space-y-6 col-start-1">
      <Heading>{t('password', 'account')}</Heading>
      <Form onSubmit={handleSubmit(onSubmit)} reset={reset} isValid={isValid && isDirty}>
        <FormControl id="password" label={t('current-password')} required>
          <Input {...register('password', { required: true })} type="password" />
        </FormControl>
        <FormControl id="new_password" label={t('new-password')} required>
          <Input {...register('new_password', { required: true })} type="password" />
        </FormControl>
      </Form>
    </section>
  )
}

function DeleteAccountConfirmModal({ show, onClose, onSuccess }) {
  const user = useAccount()
  const remove = useUsers().remove(user.id).mutate

  const onSubmit = () => {
    remove(null, { onSuccess, onError: err => console.error({ err }) })
  }

  return (
    <Modal show={show} onClose={onClose}>
      <div className="col-span-full text-center">
        <FontAwesomeIcon icon={faExclamationTriangle} className="text-warning-500 text-3xl" />
        <H4 className="mt-3">{t('account-delete-warning-heading', 'account')}</H4>
      </div>
      <div className="col-span-full text-center mt-4">
        <Text>{t('account-delete-warning-copy', 'account')}</Text>
      </div>
      <div className="col-span-full text-center mt-6 mb-4">
        <ButtonGroup>
          <Button color="error" onClick={onSubmit}>
            {t('buttons.delete-account')}
          </Button>
          <Button variant="ghost" color="error" onClick={onClose}>
            {t('buttons.cancel')}
          </Button>
        </ButtonGroup>
      </div>
    </Modal>
  )
}

function DeleteAccountSuccessModal({ show }) {
  const router = useRouter()
  const { handleSubmit, register } = useForm()

  const onSubmit = values => {
    console.log('submit frustrations, suggestions', { values })
    onClose()
  }

  const onClose = () => {
    logout()
    router.push('/')
  }

  return (
    <Modal show={show} onClose={onClose}>
      <div className="col-span-full text-center">
        <FontAwesomeIcon icon={faCheckCircle} className="text-green-500 text-3xl" />
        <H4 className="mt-3">{t('account-delete-success-heading', 'account')}</H4>
      </div>
      <div className="col-span-full text-center mt-4">
        <Text>{t('account-delete-success-copy', 'account')}</Text>
      </div>
      <form onSubmit={handleSubmit(onSubmit)} className="col-span-full mt-8">
        <FormControl id="frustrations" label={t('frustrations')} optional>
          <TextArea {...register('frustrations')} placeholder={t('placeholder.frustrations')} />
        </FormControl>
        <FormControl id="suggestions" label={t('suggestions')} optional className="mt-6">
          <TextArea {...register('suggestions')} placeholder={t('placeholder.suggestions')} />
        </FormControl>
        <div className="col-span-full text-center mt-6 mb-4">
          <ButtonGroup>
            <Button onClick={onSubmit}>{t('buttons.submit-response')}</Button>
            <Button variant="ghost" onClick={onClose}>
              {t('buttons.skip')}
            </Button>
          </ButtonGroup>
        </div>
      </form>
    </Modal>
  )
}

function AccountSettingsDeleteAccount() {
  const [showConfirm, setConfirm] = useState(false)
  const [showSuccess, setShowSuccess] = useState(false)

  return (
    <>
      <Heading>{t('delete', 'account')}</Heading>
      <Text as="p" className="mt-2">
        {t('delete-account-copy', 'account')}
      </Text>
      <div className="mt-4">
        <ButtonGroup>
          <Button variant="primary" color="error" onClick={() => setConfirm(true)}>
            {t('buttons.delete-account')}
          </Button>
        </ButtonGroup>
      </div>
      <DeleteAccountConfirmModal
        show={showConfirm}
        onClose={() => setConfirm(false)}
        onSuccess={() => setShowSuccess(true)}
      />
      <DeleteAccountSuccessModal show={showSuccess} />
    </>
  )
}

const AccountContext = createContext<User>(null)

const useAccount = () => useContext(AccountContext)

function Content() {
  const { data: user, isLoading, isError } = useMe()

  if (isLoading) return <Loader />
  if (isError)
    return (
      <Alert.Error
        title={t('errors.me.title')}
        description={t('errors.me.description')}
        alertStyle="subtle"
      />
    )

  return (
    <AccountContext.Provider value={user}>
      <article className="col-span-9 grid grid-cols-9 space-y-10">
        <div className="col-span-5">
          <AccountSettingsProfile />
        </div>
        <Divider color="light" className="col-span-9" />
        <div className="col-span-5">
          <AccountSettingsPassword />
        </div>
        <Divider color="light" className="col-span-9" />
        <div className="col-span-9">
          <AccountSettingsDeleteAccount />
        </div>
      </article>
    </AccountContext.Provider>
  )
}

export default function AccountSettings() {
  return (
    <Base>
      <TopContent heading={<Breadcrumbs path={t('settings-account', 'paths')} />} />
      <div className="col-span-9 mt-10">
        <Alert.Info alertStyle="topAccent" description={t('alert', 'account')} />
      </div>
      <div className="col-span-9 mt-8">
        <Content />
      </div>
    </Base>
  )
}
