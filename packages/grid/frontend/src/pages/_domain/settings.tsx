import { useRef, useState } from 'react'
import { Badge, Button, Divider, H5, Input, Tabs, Text } from '@/omui'
import { PlusIcon } from '@heroicons/react/solid'
import { TopContent, InputCopyToClipboard } from '@/components/lib'
import { Switch } from '@/omui/components/Switch/Switch'
import { useSettings } from '@/lib/data'
import { sections } from '@/content'
import { Base } from '@/components/Layouts'
import { useForm } from 'react-hook-form'
import { t } from '@/i18n'
import { Alert } from '@/components/Alert'
import { FormControl } from '@/omui/components/FormControl/FormControl'
import {
  DomainSettingsProvider,
  useDomainSettings,
} from '@/components/Settings/useDomainSettings'
import { Loader } from '@/components/Loader'
import { Profile } from '@/components/Settings/Profile'

const tabList = [
  { title: 'Profile', id: 0, Component: Profile },
  { title: 'Configurations', id: 1, Component: Configs, disabled: true },
  { title: 'Updates', id: 2, Component: Updates, disabled: true },
]

function Configs() {
  const { settings } = useDomainSettings()
  const update = useSettings().create(null, { multipart: true }).mutate
  const fileInputRef = useRef()
  const [file, setFile] = useState(settings.daa_document)
  const [daa, setDAA] = useState(settings!.daa)

  const onSubmit = () => {
    const formData = new FormData()
    Object.keys(settings).forEach((key) => formData.append(key, settings[key]))
    formData.append('daa_document', file.name)
    formData.append('daa', daa)
    update(formData)
  }

  return (
    <div className="col-start-3 col-span-8 space-y-8">
      <div className="space-y-3">
        <div className="flex flex-row justify-between items-center">
          <H5>{t('require-daa', 'settings')}</H5>
          <Switch size="md" checked={daa} onChange={() => setDAA(!daa)} />
        </div>
        <Text as="p" size="sm">
          {t('require-daa-copy', 'settings')}
        </Text>
      </div>
      <Divider color="light" />
      <div className="space-y-3">
        <H5 className={!daa && 'opacity-60'}>
          {t('daa', 'settings')} <Text className="text-primary-600">*</Text>
        </H5>
        <Text as="p" size="sm" className={!daa && 'opacity-60'}>
          {t('daa-copy', 'settings')}
        </Text>
        <div className="pt-3">
          {settings.daa_document ? (
            <div className="w-full grid grid-cols-8">
              <div className="col-span-5 flex space-x-2 items-center">
                <div className="bg-gray-100 w-full px-2">
                  <Text size="xs" bold>
                    {settings.daa_document === 'undefined'
                      ? 'data_agreement.pdf'
                      : settings.daa_document}
                  </Text>
                </div>
                <Text size="sm" className="flex-shrink-0 text-gray-400">
                  4 hours ago
                </Text>
              </div>
            </div>
          ) : (
            <>
              {file && (
                <div className="mb-4">
                  <Badge type="subtle" variant="gray">
                    {file?.name}
                  </Badge>
                </div>
              )}
              <Button
                size="xs"
                variant="outline"
                disabled={!daa}
                leftIcon={PlusIcon}
                onClick={() => {
                  // @ts-ignore
                  fileInputRef.current.click()
                }}
              >
                {file ? t('buttons.replace-file') : t('buttons.upload-file')}
              </Button>
              <input
                type="file"
                ref={fileInputRef}
                hidden
                onChange={(e) => setFile(e.target.files[0])}
                accept=".pdf,.ps,.doc,.docx,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              />
            </>
          )}
        </div>
      </div>
      {settings.daa_document && (
        <Alert.Base
          alertStyle="subtle"
          className="text-gray-800 bg-primary-100"
          description={<Text size="sm">{t('daa-alert', 'settings')}</Text>}
        />
      )}
      <Button
        variant="primary"
        type="button"
        disabled={daa && !file}
        onClick={onSubmit}
      >
        Save Changes
      </Button>
    </div>
  )
}

function Updates() {
  return (
    <div className="col-start-3 col-span-8 space-y-8">
      <CurrentVersion />
      <Divider color="light" />
      <UpdateVersion />
      <Button>Update</Button>
    </div>
  )
}

function CurrentVersion() {
  // const version = {updated: dayjs('2021-10-01').format('YYYY-MMM-DD HH:ss'), name: '0.6 alpha'}
  const versionInfo = [
    { label: t('last-updated'), value: '' },
    { label: t('version'), value: '' },
  ]
  return (
    <div className="space-y-3">
      <H5>{t('current-version')}</H5>
      {versionInfo.map((info) => (
        <div key={info.label} className="space-x-3">
          <Text bold size="sm">
            {info.label}:
          </Text>
          <Text mono size="sm">
            {info.value}
          </Text>
        </div>
      ))}
    </div>
  )
}

const updateVersionForm = [
  {
    col: 4,
    placeholder: 'github.com/openmined/pysyft',
    id: 'repo',
    label: 'Repository',
    optional: true,
  },
  { col: 2, placeholder: 'dev', id: 'branch', label: 'Branch', optional: true },
  {
    col: 2,
    placeholder: 'da812378asd...',
    id: 'hash',
    label: 'Hash',
    optional: true,
  },
]

function UpdateVersion() {
  const { handleSubmit, register } = useForm()

  const onSubmit = () => {}

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-3">
      <H5>{t('update-version', 'settings')}</H5>
      <Text as="p" size="sm">
        {t('update-version-copy', 'settings')}
      </Text>
      <div className="grid grid-cols-8 gap-4 text-sm">
        {updateVersionForm.map((item) => (
          <div key={item.id} className={`col-span-${item.col}`}>
            <FormControl
              id={item.id}
              label={item.label}
              optional={item.optional}
            >
              <Input placeholder={item.placeholder} {...register(item.id)} />
            </FormControl>
          </div>
        ))}
      </div>
    </form>
  )
}

export default function Settings() {
  const { data: settings, isLoading, isError } = useSettings().all()
  const [selected, setSelected] = useState(() => tabList[0].id)
  const Component = tabList[selected].Component

  return (
    <DomainSettingsProvider value={{ settings }}>
      <Base>
        <TopContent heading={sections.settings.heading} />
        <Text as="p" className="col-span-full py-4 text-gray-600">
          {sections.settings.description.replace(
            '{{domain_name}}',
            settings?.domain_name ?? ''
          )}
        </Text>
        <InputCopyToClipboard url={settings?.url} text="Copy URL" />
        <div className="col-span-full mt-10 mb-16">
          <Tabs
            align="auto"
            variant="outline"
            active={selected}
            tabsList={tabList}
            onChange={setSelected}
          />
          <div className="grid grid-cols-12 gap-4 pt-16">
            {isLoading && <Loader />}
            {isError && (
              <Alert.Error
                alertStyle="subtle"
                title={t('error-loading-settings')}
              />
            )}
            {settings && <Component />}
          </div>
        </div>
      </Base>
    </DomainSettingsProvider>
  )
}
