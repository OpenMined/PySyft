import { useEffect, useState } from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Text } from '@/omui'
import { TopContent, SearchInput } from '@/components/lib'
import { NetworkAccordion } from '@/components/NetworkAccordion'
import { sections } from '@/content'
import { SingleCenter } from '@/components/Layouts'
import ky from 'ky'

export default function Networks() {
  const [networks, setNetworks] = useState([])
  useEffect(() => {
    const getNetworks = async () => {
      const networkList = await ky
        .get('https://raw.githubusercontent.com/OpenMined/NetworkRegistry/main/networks.json')
        .json()
      setNetworks(networkList?.networks ?? [])
    }
    getNetworks()
  }, [])

  return (
    <SingleCenter>
      <TopContent
        icon={() => <FontAwesomeIcon icon={sections.networks.icon} className="text-3xl" />}
        heading={sections.networks.heading}
      />
      <Text as="p" className="col-span-full mt-8 text-gray-600">
        {sections.networks.description}
      </Text>
      <div className="mt-12 col-span-3">
        <SearchInput />
      </div>
      <div className="col-span-full mt-10">
        <NetworkAccordion networks={networks} />
      </div>
    </SingleCenter>
  )
}
