import { useEffect, useState } from 'react'
import {
  Badge,
  Button,
  Divider,
  H4,
  H5,
  Input,
  Tag,
  Text,
  TextArea,
} from '@/omui'
import { XIcon } from '@heroicons/react/solid'
import { Optional, ButtonGroup } from '@/components/lib'
import { useSettings } from '@/lib/data'
import { formatDate } from '@/utils'
import {
  Controller,
  FormProvider,
  useForm,
  useFormContext,
} from 'react-hook-form'
import { t } from '@/i18n'
import { useDisclosure } from '@/hooks/useDisclosure'
import Modal from '@/components/Modal'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  faCheckCircle,
  faExclamationTriangle,
} from '@fortawesome/free-solid-svg-icons'
import { FormControl } from '@/omui/components/FormControl/FormControl'
import { logout } from '@/lib/auth'
import { useRouter } from 'next/router'
import { useDomainSettings } from './useDomainSettings'

function General() {
  const { settings } = useDomainSettings()

  const information = [
    { name: t('domain-name'), value: settings?.domain_name },
    {
      name: 'ID#',
      value: (
        <span className="dark">
          <Badge variant="gray" type="subtle">
            {settings?.node_id}
          </Badge>
        </span>
      ),
    },
    { name: t('hosted-datasets'), value: settings?.total_datasets },
    { name: t('deployed-on'), value: formatDate(settings?.deployed_on) },
    { name: t('owner'), value: settings?.owner },
  ]

  return (
    <div className="w-full space-y-3 relative">
      <H5>{t('general', 'settings')}</H5>
      {information.map((info) => {
        return (
          <div key={info.name} className="space-x-3">
            <Text bold size="sm">
              {info.name}:
            </Text>
            {typeof info.value === 'string' ? (
              <Text mono>{info.value}</Text>
            ) : (
              info.value
            )}
          </div>
        )
      })}
    </div>
  )
}

function DomainDescription() {
  const { register } = useFormContext()
  return (
    <div className="space-y-3">
      <H5>
        {t('domain-description', 'settings')}
        <Optional />
      </H5>
      <TextArea
        placeholder={t('placeholder.describe-your-domain')}
        rows="5"
        {...register('description')}
      />
    </div>
  )
}

function SupportEmail() {
  const { register } = useFormContext()
  return (
    <div className="space-y-3">
      <H5>
        {t('support-email', 'settings')} <Optional />
      </H5>
      <FormControl hint={t('support-email-label', 'settings')} id="contact">
        <Input placeholder="support@company.org" {...register('contact')} />
      </FormControl>
    </div>
  )
}

function Tags() {
  const { control, setValue } = useFormContext()
  const { settings } = useDomainSettings()
  const [tags, setTags] = useState(() =>
    Array.isArray(settings?.tags) ? settings.tags : []
  )

  const updateTags = (tag) => {
    if (!tags.includes(tag)) setTags((prev) => [...prev, tag])
  }

  useEffect(() => {
    setValue('tags', tags)
  }, [tags])

  return (
    <div className="space-y-3">
      <H5>
        {t('tags', 'settings')}
        <Optional />
      </H5>
      <Controller
        control={control}
        name="thisTag"
        render={({ field }) => {
          const { onChange, onBlur, value } = field
          return (
            <Input
              placeholder={t('placeholder.create-new-tag')}
              addonRight="Add"
              value={value}
              onChange={onChange}
              onBlur={onBlur}
              addonRightProps={{
                onClick: () => {
                  updateTags(value)
                  field.onChange('')
                },
                className: 'cursor-pointer',
              }}
            />
          )
        }}
      />
      {tags &&
        [...tags].map((entry) => (
          <Tag
            tagType="round"
            variant="gray"
            size="sm"
            className="mr-2 cursor-pointer"
            key={entry}
            icon={XIcon}
            iconSide="right"
            onClick={(e) =>
              setTags((prev) => prev.filter((tag) => tag !== entry))
            }
          >
            {entry}
          </Tag>
        ))}
    </div>
  )
}

function Profile() {
  const { settings } = useDomainSettings()
  const { mutate: update, isLoading } = useSettings().create(null, {
    multipart: true,
  })
  const methods = useForm({
    mode: 'onChange',
    defaultValues: {
      description: settings?.description,
      tags: settings?.tags,
      contact: settings?.contact,
    },
  })

  const onSubmit = ({ thisTag, ...values }) => {
    const formData = new FormData()
    formData.append('settings', JSON.stringify({ ...values }))
    formData.append('file', new Blob())
    update(formData)
  }

  return (
    <div className="col-start-3 col-span-8 space-y-8">
      <General />
      <Divider color="light" />
      <FormProvider {...methods}>
        <form onSubmit={methods.handleSubmit(onSubmit)} className="space-y-8">
          <DomainDescription />
          <SupportEmail />
          <Tags />
          <Button type="submit" isLoading={isLoading}>
            {t('buttons.save-changes')}
          </Button>
        </form>
      </FormProvider>
      <Divider color="light" />
      <ResetNode isLoading={isLoading} />
    </div>
  )
}

function ResetNodeConfirmationModal({ show, onClose, onSuccess, isLoading }) {
  const onSubmit = () => {
    onSuccess()
  }
  return (
    <Modal show={show} onClose={onClose}>
      <div className="col-span-full text-center">
        <FontAwesomeIcon
          icon={faExclamationTriangle}
          className="text-warning-500 text-3xl"
        />
        <H4 className="text-gray-800 mt-3">
          {t('reset-confirmation-heading', 'settings')}
        </H4>
      </div>
      <div className="col-span-full text-center mt-4">
        <Text size="sm">{t('reset-confirmation-copy', 'settings')}</Text>
      </div>
      <div className="col-span-full text-center mt-6 mb-4">
        <ButtonGroup>
          <Button color="error" onClick={onSubmit} isLoading={isLoading}>
            {t('buttons.reset-purge-node')}
          </Button>
          <Button
            variant="ghost"
            color="error"
            onClick={onClose}
            disabled={isLoading}
          >
            {t('buttons.cancel')}
          </Button>
        </ButtonGroup>
      </div>
    </Modal>
  )
}

function ResetNodeSuccessModal({ show, isLoading }) {
  const router = useRouter()
  const { handleSubmit, register } = useForm()

  const onSubmit = (values) => {
    console.log('submit frustrations, suggestions', { values })
    onClose()
  }

  const onClose = () => {
    logout()
    router.push('/login')
  }

  return (
    <Modal show={show} onClose={onClose}>
      <div className="col-span-full text-center">
        <FontAwesomeIcon
          icon={faCheckCircle}
          className="text-green-500 text-3xl"
        />
        <H4 className="mt-3">{t('reset-success-heading', 'settings')}</H4>
      </div>
      <div className="col-span-full text-center mt-4">
        <Text>{t('reset-success-copy', 'settings')}</Text>
      </div>
      <form onSubmit={handleSubmit(onSubmit)} className="col-span-full mt-8">
        <FormControl id="frustrations" label={t('frustrations')} optional>
          <TextArea
            {...register('frustrations')}
            placeholder={t('placeholder.frustrations')}
          />
        </FormControl>
        <FormControl
          id="suggestions"
          label={t('suggestions')}
          optional
          className="mt-6"
        >
          <TextArea
            {...register('suggestions')}
            placeholder={t('placeholder.suggestions')}
          />
        </FormControl>
        <div className="col-span-full text-center mt-6 mb-4">
          <ButtonGroup>
            <Button onClick={onSubmit} isLoading={isLoading}>
              {t('buttons.submit-response')}
            </Button>
            <Button variant="ghost" onClick={onClose} disabled={isLoading}>
              {t('buttons.skip')}
            </Button>
          </ButtonGroup>
        </div>
      </form>
    </Modal>
  )
}

function ResetNode({ isLoading }) {
  const { open, isOpen, close } = useDisclosure(false)
  const { open: openSuccess, isOpen: isOpenSuccess } = useDisclosure(false)
  return (
    <div className="w-full">
      <H5>{t('reset', 'settings')}</H5>
      <Text as="p" className="mt-4">
        {t('reset-copy', 'settings')}
      </Text>
      <Button
        className="mt-8 bg-error-500"
        variant="primary"
        type="button"
        onClick={open}
        isLoading={isLoading}
      >
        {t('buttons.reset-purge-node')}
      </Button>
      <ResetNodeConfirmationModal
        show={isOpen}
        onClose={close}
        onSuccess={openSuccess}
        isLoading={isLoading}
      />
      <ResetNodeSuccessModal show={isOpenSuccess} isLoading={isLoading} />
    </div>
  )
}

export { Profile }
