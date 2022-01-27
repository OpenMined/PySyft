import Link from 'next/link'
import tw, { styled } from 'twin.macro'
import { Popover } from '$components/Popover'
import { Logo } from '$components/Logo'
import { Divider } from '$components/Divider'

const Panel = styled.div`
  ${tw`flex flex-col`}
  ${tw`py-2.5 px-4`}
  ${tw`bg-gray-50 shadow-domain-menu`}
`

const NodeInfo = ({ domain_name = 'Unnamed Domain', status = 'Online' }) => {
  const DomainContent = () => (
    <div tw="flex flex-col">
      <span tw="font-bold text-gray-800">{domain_name}</span>
      <span tw="block text-gray-600 italic">{status}</span>
    </div>
  )

  return (
    <div tw="flex gap-4 h-16 items-center">
      <Logo lockup="mark" color="light" product="pygrid" />
      <DomainContent />
    </div>
  )
}

const items = [
  { label: 'Profile', href: '/settings?tab=profile' },
  { label: 'Configurations', href: '/settings?tab=configurations' },
  { label: 'Version Updates', href: '/settings?tab=version' },
]

const Menu = () => (
  <Panel>
    <NodeInfo />
    <Divider />
    {items.map(({ label, href }) => (
      <Link key={label} href={href} passHref>
        <ParentItem>
          <a>{label}</a>
        </ParentItem>
      </Link>
    ))}
    <Divider />
    <ParentItem>
      <button>Logout</button>
    </ParentItem>
  </Panel>
)

const ParentItem = styled.div`
  ${tw`flex items-center py-1 px-4`}
  ${tw`text-gray-800 hover:(text-gray-0 bg-primary-500) disabled:opacity-50`}
  ${tw`cursor-pointer`}
`
export const DomainMenu = () => (
  <Popover>
    <Menu />
  </Popover>
)
