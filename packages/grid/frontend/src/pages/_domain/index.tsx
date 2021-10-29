import {useState} from 'react'
import {Page} from '@/components/Page'
import {Grid} from '@/components/Grid'
import {QuickNav} from '@/components/QuickNav'
import {Badge, Button, Divider, H2, H5, Input, Tabs, Tag, Text, TextArea} from '@/omui'
import dayjs from 'dayjs'
import type {ReactNode} from 'react'
import {XIcon} from '@heroicons/react/solid'

const TopContent = ({heading, children}: {heading: string; children: ReactNode}) => (
  <div className="col-span-full">
    <div className="flex justify-between">
      <H2>{heading}</H2>
      <QuickNav />
    </div>
    {children}
  </div>
)

const CopyURL = ({url}: {url: string}) => (
  <div style={{width: 368}}>
    <Input variant="outline" addonRight={<Text size="sm">Copy URL</Text>} defaultValue={url} />
  </div>
)
const HeaderDescription = props => <Text as="p" className="py-4 text-gray-600" {...props}></Text>

const tabList = [
  {title: 'Profile', id: 1},
  {title: 'Configurations', id: 2},
  {title: 'Updates', id: 3}
]

const information = [
  {name: 'Domain Name', property: 'domain_name'},
  {
    name: 'ID#',
    property: 'id',
    Component: props => (
      <span className="dark">
        <Badge {...props} variant="gray" type="subtle" />
      </span>
    )
  },
  {name: 'Hosted datasets', property: 'num_datasets'},
  {name: 'Deployed on', property: 'created_at'},
  {name: 'Owner', property: 'owner'}
]

const tabs = ['Commodities', 'Trade', 'Canada']

export default function Settings() {
  const [selected, setSelected] = useState(() => tabList[0].id)
  const domainInformation = {
    domain_name: 'Canada Domain',
    id: '449f4f997a96467f90f7af8b396928f1',
    num_datasets: 2,
    created_at: dayjs('2021-07-09').format('YYYY-MMM-DD HH:ss'),
    owner: 'Kyoko Eng'
  }

  return (
    <Page>
      <Grid>
        <TopContent heading="Domain Settings">
          <HeaderDescription>
            Provide contextual information for the Canada Domain node and set structural configurations.
          </HeaderDescription>
          <CopyURL url="domain-specific-url.com" />
        </TopContent>
        <div className="col-span-full mt-10 mb-16">
          <Tabs align="auto" variant="outline" active={selected} tabsList={tabList} onChange={id => setSelected(id)} />
        </div>
        <div className="w-full col-start-3 col-span-8 relative space-y-3">
          <H5>General</H5>
          {information.map(info => (
            <div key={info.name} className="space-x-3">
              <Text bold size="sm">
                {info.name}:
              </Text>
              {info.Component ? (
                <info.Component>{domainInformation[info.property]}</info.Component>
              ) : (
                <Text mono>{domainInformation[info.property]}</Text>
              )}
            </div>
          ))}
          <div className="absolute w-full h-full bottom-0 right-0">
            <Button className="absolute right-0 bottom-0 bg-error-500" size="xs" variant="primary">
              Delete Node
            </Button>
          </div>
        </div>
        <Divider color="light" className="col-span-8 col-start-3 my-8" />
        <div className="w-full col-start-3 col-span-8 relative space-y-8">
          <div className="space-y-3">
            <H5>
              Domain Description <Text className="text-primary-600 italic pl-2">Optional</Text>
            </H5>
            <TextArea placeholder="Describe your domain to potential users here..." rows="5" />
          </div>
          <div className="space-y-3">
            <H5>
              Support Email <Text className="text-primary-600 italic pl-2">Optional</Text>
            </H5>
            <Input placeholder="support@company.org" />
          </div>
          <div className="space-y-3">
            <H5>
              Tags <Text className="text-primary-600 italic pl-2">Optional</Text>
            </H5>
            <Input
              type="textarea"
              placeholder="Create new tag here..."
              addonRight="Add"
              addonRightProps={{onClick: () => console.log('clicked add'), className: 'cursor-pointer'}}
            />
            {tabs.map(entry => (
              <Tag
                tagType="round"
                variant="gray"
                size="sm"
                className="mr-2 cursor-pointer"
                key={entry}
                icon={XIcon}
                iconSide="right"
                onClick={e => console.log({tag: e.target.value})}>
                {entry}
              </Tag>
            ))}
          </div>
        </div>
        <div className="col-start-3 col-span-8 mt-8">
          <Button variant="primary" type="submit">
            Save Changes
          </Button>
        </div>
      </Grid>
    </Page>
  )
}
