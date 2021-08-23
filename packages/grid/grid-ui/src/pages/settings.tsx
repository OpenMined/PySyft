import {Page} from '@/components'
import {useSettings} from '@/lib/data'

export default function Settings() {
  const {data: settings} = useSettings()

  return (
    <Page
      title="Domain Settings"
      description={`Set structural and usage configurations for ${
        settings?.domainName ?? settings?.nodeName ?? '...'
      }`}>
      <section className="max-w-lg space-y-2 lg:max-w-3xl">
        <h3 className="font-medium">Configurations</h3>
        <p>This section is still under construction.</p>
      </section>
      {/* TODO: uncomment this when /setup is fixed
      <section className="flex justify-between space-x-6">
        {settings ? <SettingsList settings={settings} /> : <SpinnerWithText>Loading Domain settings</SpinnerWithText>}
      </section>
      */}
    </Page>
  )
}
