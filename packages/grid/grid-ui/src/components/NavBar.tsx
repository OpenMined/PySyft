import {Fragment, createContext, useContext, useCallback, useState} from 'react'
import Link from 'next/link'
import {useRouter} from 'next/router'
import cn from 'classnames'
import {Dialog, Transition} from '@headlessui/react'
import {MenuIcon, XIcon} from '@heroicons/react/solid'
import {DomainConnectionStatus} from '@/components'
import {useMe, useRequests, useDomainStatus} from '@/lib/data'
import {useAuth} from '@/context/auth-context'

const navigation = [
  {name: 'Datasets', href: '/datasets', disabled: true},
  {name: 'Models', href: '/models', disabled: true},
  {name: 'Requests', href: '/requests', disabled: true},
  {name: 'Tensors', href: '/tensors', disabled: true},
  {name: 'Users', href: '/users'},
  {name: 'Roles & Permissions', href: '/permissions'},
  {name: 'Dashboard', href: '/dashboard', disabled: true},
  {name: 'Networks', href: '#', disabled: true}, //networks'},
  {name: 'Settings', href: '/settings', disabled: true}
]

const SidebarContext = createContext(null)

function MobileSidebarMenuContent() {
  return (
    <div className="relative flex flex-col flex-1 w-full text-gray-800 bg-gray-100">
      <div className="flex-1 h-0 pt-5 pb-4 overflow-y-auto">
        <Link href="/">
          <a>
            <NodeInfo />
          </a>
        </Link>
        <DomainConnectionStatus />
        <Navigation />
      </div>
      <LogoutBox />
    </div>
  )
}

function MobileSidebarDisplay() {
  const {open} = useContext(SidebarContext)
  return (
    <div className="flex items-center justify-between w-full h-20 border cursor-pointer bg-blueGray-50 md:hidden">
      <Link href="/">
        <a>
          <NodeInfo />
        </a>
      </Link>
      <button onClick={() => open(true)}>
        <p className="sr-only">Press to open the navigation menu</p>
        <MenuIcon className="w-8 h-8 mx-4" role="presentation" />
      </button>
    </div>
  )
}

function MobileSidebar() {
  const {isOpen, open} = useContext(SidebarContext)

  return (
    <>
      <MobileSidebarDisplay />
      <Transition.Root show={isOpen} as={Fragment}>
        <Dialog as="div" static className="fixed inset-0 z-40 flex md:hidden" open={isOpen} onClose={open}>
          <Transition.Child
            as={Fragment}
            enter="transition-opacity ease-linear duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="transition-opacity ease-linear duration-300"
            leaveFrom="opacity-100"
            leaveTo="opacity-0">
            <Dialog.Overlay className="fixed inset-0 w-full bg-gray-600 bg-opacity-90" />
          </Transition.Child>
          <div className="flex-shrink-0 w-14" aria-hidden="true" />
          <Transition.Child
            as={Fragment}
            enter="transition ease-in-out duration-300 transform"
            enterFrom="translate-x-full"
            enterTo="translate-x-0"
            leave="transition ease-in-out duration-300 transform"
            leaveFrom="translate-x-0"
            leaveTo="translate-x-full">
            <MobileSidebarMenuContent />
          </Transition.Child>
          <Transition.Child
            as={Fragment}
            enter="ease-in-out duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="ease-in-out duration-300"
            leaveFrom="opacity-100"
            leaveTo="opacity-0">
            <div className="absolute top-0 left-0 pt-2 -mr-12">
              <button
                className="flex items-center justify-center w-10 h-10 mr-1 rounded-full focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
                onClick={() => open(false)}>
                <span className="sr-only">Close menu</span>
                <XIcon className="w-6 h-6 text-white" aria-hidden="true" />
              </button>
            </div>
          </Transition.Child>
        </Dialog>
      </Transition.Root>
    </>
  )
}

function DesktopSidebar() {
  return (
    <aside className="sticky top-0 hidden h-screen border border-gray-200 md:flex">
      <div className="flex flex-shrink-0 h-full text-gray-800 bg-blueGray-100">
        <div className="flex flex-col w-64">
          <div className="flex flex-col flex-1 h-0">
            <SidebarContent />
          </div>
        </div>
      </div>
    </aside>
  )
}

function NodeInfo() {
  const {data} = useDomainStatus()
  const domainName = data?.nodeName

  return (
    <div className="flex items-center flex-shrink-0 px-4">
      <img className="w-auto h-16" src="/assets/logo.png" alt="PyGrid Domain" />
      <span className="text-xl">{domainName}</span>
    </div>
  )
}

function LogoutBox() {
  const router = useRouter()
  const {logout} = useAuth()
  const {data: me} = useMe()

  const doLogout = useCallback(() => {
    logout()
    router.push('/login')
  }, [logout, router])

  return (
    <button onClick={doLogout}>
      <div className="flex flex-shrink-0 p-4 border-t border-gray-200 group">
        <div className="flex-shrink-0 block w-full cursor-pointer">
          <div className="flex items-center">
            <div className="text-left font-regular">
              <p className="text-sm text-gray-500 transition-colors">{me?.email}</p>
              <p className="text-xs group-hover:text-blue-500">Log out</p>
            </div>
          </div>
        </div>
      </div>
    </button>
  )
}

function Navigation() {
  const {current} = useContext(SidebarContext)
  const {all} = useRequests()
  const {data: requests} = all()
  const totalRequests = requests?.filter(request => request.status === 'pending').length ?? null

  return (
    <nav className="px-2 mt-5 space-y-1">
      {navigation.map(item =>
        item.disabled ? (
          <span
            className={cn(
              'text-gray-400 cursor-default',
              'flex items-center px-2 py-2 text-sm font-regular rounded-sm'
            )}>
            {item.name}
          </span>
        ) : (
          <Link href={item.href} key={item.href}>
            <a
              className={cn(
                item.href === current
                  ? 'bg-cyan-500 text-white'
                  : 'text-gray-800 hover:text-white hover:bg-sky-600 hover:bg-opacity-75 active:bg-opacity-100',
                'group flex items-center px-2 py-2 text-sm font-regular rounded-sm'
              )}>
              {item.name} {item.href === '/requests' && totalRequests > 0 && `(${totalRequests})`}
            </a>
          </Link>
        )
      )}
    </nav>
  )
}

function SidebarContent() {
  return (
    <>
      <div className="flex flex-col flex-1 pt-2 pb-4 overflow-y-auto">
        <Link href="/">
          <a>
            <NodeInfo />
          </a>
        </Link>
        <Navigation />
      </div>
      <DomainConnectionStatus />
      <LogoutBox />
    </>
  )
}

export function Sidebar() {
  const [isOpen, open] = useState(false)
  const router = useRouter()
  return (
    <SidebarContext.Provider value={{isOpen, open, current: router.route}}>
      <MobileSidebar />
      <DesktopSidebar />
    </SidebarContext.Provider>
  )
}
