import { createContext, useContext } from 'react'
import { Badge, Button, Divider, Text } from '@/omui'
import { Dot } from '@/components/lib'
import { Accordion } from '@/components/Accordion'
import { formatDate } from '@/utils'
import { useAssociationRequest } from '@/lib/data'
import { t } from '@/i18n'
import api from '@/utils/api'

import type { Network } from '@/types/network'

const SLACK_ICON = 'https://upload.wikimedia.org/wikipedia/commons/7/76/Slack_Icon.png'

const NetworkContext = createContext({ network: null })

interface AccordionNetwork {
  networks: Array<Network>
}

export function NetworkAccordion({ networks = [] }: AccordionNetwork) {
  if (!networks.length) return null
  return (
    <Accordion>
      {networks.map(network => (
        <NetworkAccordionItem key={network.id ?? network.host_or_ip} {...network} />
      ))}
    </Accordion>
  )
}

function NetworkAccordionItem(network: Network) {
  return (
    <NetworkContext.Provider value={{ network }}>
      <Accordion.Item>
        <Title />
        <Panel />
      </Accordion.Item>
    </NetworkContext.Provider>
  )
}

function Title() {
  const { network } = useContext(NetworkContext)
  const create = useAssociationRequest().create().mutate

  return (
    <div className="flex w-full justify-between">
      <div className="flex space-x-4 items-center">
        <Text bold size="lg">
          {network?.name}
        </Text>
        {network?.status === 'guest' ? (
          <Badge variant="primary" type="subtle">
            {t('guest')}
          </Badge>
        ) : (
          <Text size="sm" className="italic">
            {t('not-a-member')}
          </Text>
        )}
      </div>
      {network.joined_on ? (
        <div className="flex items-center space-x-2">
          <div>
            <Text size="sm">{formatDate(network.joined_on)}</Text>
            <Dot />
          </div>
          <Button
            variant="outline"
            size="sm"
            type="button"
            className="text-error-600 border-error-500 hover:bg-error-500"
            onClick={e => {
              e.stopPropagation()
              create({ type: 'remove_network_connection' })
            }}
          >
            {t('buttons.leave-network')}
          </Button>
        </div>
      ) : (
        <Button
          variant="outline"
          size="sm"
          type="button"
          onClick={async e => {
            e.stopPropagation()
            await api.post(`vpn/join/docker-host:9081`).json()
            api.post(`association-requests/request?target=docker-host:9081&source=nesio`)
            // await api.post(`vpn/join/${network?.vpn_host_or_ip}`).json()
            // api.post(`association-requests/request?target=${network?.host_or_ip}:${network?.port}&source=nesio`)
          }}
        >
          {t('buttons.join-as-guest')}
        </Button>
      )}
    </div>
  )
}

function Panel() {
  const { network } = useContext(NetworkContext)

  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="px-6 border border-gray-100 py-4 rounded">
        <div className="space-y-3">
          <div className="flex space-x-2 items-center">
            <Text as="p" bold size="sm">
              {t('host')}
            </Text>
            <Badge variant="gray" type="subtle">
              {network?.host_or_ip}
            </Badge>
          </div>
          <div className="flex space-x-2 items-center">
            <Text as="p" bold size="sm">
              {t('VPN')}
            </Text>
            <Badge variant="primary" type="subtle">
              {network?.vpn_host_or_ip}
            </Badge>
          </div>
          <div className="flex space-x-2 items-center">
            <Text as="p" bold size="sm">
              {t('website')}
            </Text>
            <a href={network?.website} target="noreferrer noopener _blank">
              <Text size="sm" underline>
                {network?.website}
              </Text>
            </a>
          </div>
          <div className="flex space-x-2 items-center">
            <Text as="p" bold size="sm">
              {t('support-email')}
            </Text>
            <a href={`mailto:${network?.admin_email}`}>
              <Text size="sm" underline>
                {network?.admin_email}
              </Text>
            </a>
          </div>
          {network?.description && (
            <>
              <div className="pt-2">
                <Divider color="light" className="my-0" />
              </div>
              <Text as="p" size="sm">
                {network.description}
              </Text>
            </>
          )}
        </div>
      </div>
      <div className="p-4">
        <a href={network?.slack} target="noopener noreferrer _blank">
          <Text as="p" size="sm" underline>
            {network?.slack}
          </Text>
        </a>
        <div className="flex items-center space-x-2">
          <img className="flex-shrink-0 w-4 h-4" src={SLACK_ICON} />
          <Text as="p" size="sm">
            {network.slack_channel}
          </Text>
        </div>
      </div>
    </div>
  )
}
