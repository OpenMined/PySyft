import tw from 'twin.macro'

interface DomainStatusProps {
  status: 'online' | 'offline' | 'loading'
}

export const DomainStatus = ({ status }: DomainStatusProps) => (
  <span tw="inline-flex items-center">
    <span css={[tw`inline-block w-1.5 h-1.5 mr-2 rounded-full`, statusColor[status]]} />
    <span>{states[status]}</span>
  </span>
)

const statusColor = {
  online: tw`bg-success-500`,
  offline: tw`bg-danger-500`,
  loading: tw`bg-gray-600`,
}

const states = {
  online: 'Domain Online',
  offline: 'Domain Offline',
  loading: 'Checking Connection',
}
