import { useState } from 'react'
import { useQueryClient, useMutation } from 'react-query'
import { NormalButton, Input } from '@/components'
import { cacheKeys } from '@/utils'
import api from '@/utils/api-axios'
import type { Settings } from '@/types/grid-types'

function ChangeNodeName({ domainName }: Partial<Settings>) {
  const queryClient = useQueryClient()
  const [newNodeName, chooseNewNodeName] = useState<string>(domainName)
  const invalidate = () => queryClient.invalidateQueries([cacheKeys.settings, cacheKeys.status])
  const mutation = useMutation(
    () => api.put(`${cacheKeys.settings}`, { domain_name: newNodeName }),
    {
      onSuccess: invalidate,
    }
  )

  return (
    <div className="flex max-w-xl space-x-4">
      <Input
        container="w-full"
        id="node-name"
        label="Change name"
        value={newNodeName}
        onChange={e => chooseNewNodeName(e.target.value)}
      />
      <NormalButton
        onClick={() => mutation.mutate()}
        className="flex-shrink-0 w-24 mt-auto"
        disabled={mutation.isLoading}
        isLoading={mutation.isLoading}
      >
        Submit
      </NormalButton>
    </div>
  )
}

interface SettingsAsProp {
  settings: Settings
}

export function SettingsList({ settings }: SettingsAsProp) {
  return (
    <div className="w-full">
      <ChangeNodeName {...settings} />
    </div>
  )
}
