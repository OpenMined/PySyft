import cn from 'classnames'
import {Divider, List, ListAvatarItem, ListFAIconItem, Text, Badge} from '@/omui'
import Link from 'next/link'
import {useRouter} from 'next/router'
import {DomainStatus} from './DomainStatus'
import {faUsers, faCheck, faLemon, faHandsHelping, faChevronDown, faUserCircle} from '@fortawesome/free-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {useMe, useSettings} from '@/lib/data'
import {t} from '@/i18n'
import {logout} from '@/lib/auth'

const navItems = [
  {name: 'Users', link: '/users', icon: faUsers},
  {name: 'Permissions', link: '/permissions', icon: faCheck},
  {
    name: 'Requests',
    link: '/requests/data',
    icon: faLemon,
    children: [
      {name: 'Data Requests', link: '/requests/data'},
      {name: 'Upgrade Requests', link: '/requests/upgrade'}
    ]
  }
]

const SidebarNav = () => {
  const router = useRouter()
  const currRoute = navItems.find(item => router.asPath.startsWith(item.link))

  return (
    <aside className="flex flex-col justify-between h-screen sticky top-0 dark pt-8 pb-6" style={{minWidth: 270}}>
      <DomainInfo />
      <Divider className="mt-6 mb-10" />
      <nav className="flex-grow overflow-auto">
        <List size="lg">
          {navItems.map(navItem => {
            const isSelected = navItem.children?.find(subItem => router.asPath.startsWith(subItem.link))
            return (
              <div key={navItem.link}>
                <Link href={navItem.link}>
                  <a>
                    <ListFAIconItem
                      key={navItem.link}
                      icon={navItem.icon}
                      iconColor="text-white"
                      className={cn(
                        'hover:bg-gray-800 hover:no-underline',
                        isSelected && 'bg-gray-800 text-gray-200',
                        !isSelected && currRoute === navItem && 'bg-gray-800 text-white'
                      )}
                    >
                      {navItem.name}
                    </ListFAIconItem>
                  </a>
                </Link>
                {isSelected && (
                  <div className="bg-gray-800 text-gray-200 py-2">
                    {navItem.children.map(subItem => {
                      const currSubRoute = router.asPath.startsWith(subItem.link)
                      return (
                        <Link key={subItem.link} href={subItem.link}>
                          <a className="group">
                            <ListFAIconItem
                              key={navItem.link}
                              icon={navItem.icon}
                              iconColor={cn(currSubRoute ? 'text-gray-700' : 'text-gray-800 group-hover:text-gray-700')}
                              className={cn(
                                'hover:bg-gray-700 hover:no-underline',
                                currSubRoute && 'bg-gray-700 text-white'
                              )}
                            >
                              {subItem.name}
                            </ListFAIconItem>
                          </a>
                        </Link>
                      )
                    })}
                  </div>
                )}
              </div>
            )
          })}
        </List>
      </nav>
      <footer className="mt-auto flex-shrink-0 space-y-3">
        <div className="w-full px-3 pb-2 text-gray-200">
          <DomainStatus textSize="md" />
        </div>
        <Divider />
        <CurrentUser />
      </footer>
    </aside>
  )
}

function CurrentUser() {
  const {data: currentUser} = useMe()

  return (
    <Link href="/account">
      <List className="text-gray-200 cursor-pointer" size="lg">
        <ListFAIconItem className="text-3xl" icon={faUserCircle}>
          {currentUser?.name}
        </ListFAIconItem>
      </List>
    </Link>
  )
}

function DomainInfo() {
  const {data: domainData} = useSettings().all()

  return (
    <List size="xl" className="px-3 flex-shrink-0">
      <ListAvatarItem src="/assets/small-grid-symbol-logo.png">
        <div className="flex flex-grow space-x-3 w-full">
          <div className="flex flex-col space-y-2 flex-grow">
            <Text bold>{domainData?.domain_name}</Text>
            <Badge variant="gray" type="solid" className="w-20" truncate>
              ID#{domainData?.node_id}
            </Badge>
            <button className="text-left" onClick={logout}>
              <Text size="sm" underline className="lowercase bg-transparent hover:text-white">
                {t('logout')}
              </Text>
            </button>
          </div>
          <div className="text-gray-400 cursor-pointer flex-shrink-0">
            <FontAwesomeIcon icon={faChevronDown} title="Open the domain configuration menu" />
          </div>
        </div>
      </ListAvatarItem>
    </List>
  )
}

export {SidebarNav}
