import React from 'react'
import tw, { styled } from 'twin.macro'
import { useRouter } from 'next/router'
import { DomainMenu } from '$components/DomainMenu'
import { Logo } from '$components/Logo'
import { Id } from '$components/Id'
import { Divider } from '$components/Divider'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import EpsilonIcon from '$icons/Epsilon'

const Container = styled.div`
  ${tw`grid pt-8 pb-6 text-gray-200`}
  grid-template-rows: max-content auto 1fr auto max-content;
`

const Top = styled.section`
  ${tw`flex flex-row gap-3 mb-6 px-3`}
  img {
    ${tw`w-12 h-12`}
  }
`

const NodeInfo = ({ domain_name = 'Canada Domain', domain_id = 'asodfu08uadf' }) => (
  <div tw="flex flex-col">
    <span tw="font-bold my-2">{domain_name}</span>
    <Id variant="solid" mode="dark" id={domain_id} />
    <button tw="text-left text-sm underline mt-4">logout</button>
  </div>
)

const MenuGroups = styled.div``

const MenuItems = styled.nav``

const MenuContent = styled.div`
  ${tw`flex flex-col mt-10 px-1`}
`

const sections = [
  { title: 'Dashboard', icon: 'th-large', href: '/dashboard', group: [] },
  { title: 'Users', icon: 'users', href: '/users', group: [] },
  { title: 'Permissions', icon: 'check', href: '/permissions', group: [] },
  {
    title: 'Requests',
    icon: <EpsilonIcon height="24" />,
    href: '/requests',
    group: [{ title: 'Data Requests' }, { title: 'Upgrade Requests' }],
  },
  { title: 'Networks', icon: 'hands-helping', href: '/networks', group: [] },
]

const NavItem = styled.nav`
  ${tw`h-16 flex items-center cursor-pointer`}
  ${tw`hover:(bg-gray-800)`}
  ${tw`active:(bg-gray-800)`}

  .icon {
    ${tw`w-10 h-10 text-lg inline-flex items-center justify-center text-gray-0`}
  }
`

export const SideMenu = () => {
  const router = useRouter()
  const { route } = router

  return (
    <Container>
      <Top>
        <Logo lockup="mark" product="pygrid" color="light" />
        <NodeInfo />
        <DomainMenu />
      </Top>
      <Divider mode="dark" />
      <MenuContent>
        {sections.map(({ title, icon, href }) => (
          <NavItem key={title} tw="inline-flex" css={[href === route && tw`bg-gray-800`]}>
            <div className="icon">
              {React.isValidElement(icon) && typeof icon !== 'string' ? (
                icon
              ) : (
                <FontAwesomeIcon icon={icon} />
              )}
            </div>
            {title}
          </NavItem>
        ))}
      </MenuContent>
      <Divider mode="dark" />
      <Footer />
    </Container>
  )
}

const Footer = styled.footer`
  ${tw`flex flex-row justify-between items-center`}
  font-size: 0.8rem;
  color: #fff;
`
