import cn from 'classnames'
import { QuickNav } from './QuickNav'
import { Badge, H2, Icon, Input, Tag, Text } from '@/omui'
import { Tooltip } from 'react-tippy'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faInfoCircle, faSearch } from '@fortawesome/free-solid-svg-icons'

import type { User } from '@/types/user'
import type { ReactNode } from 'react'

import commonStrings from '@/i18n/en/common.json'

const Optional = (props) => (
  <Text
    {...props}
    className={cn('text-primary-600 italic pl-1', props?.className)}
  >
    Optional
  </Text>
)

const InputCopyToClipboard = ({ url, text }: { url: string; text: string }) => (
  <div style={{ width: 368 }}>
    <Input
      variant="outline"
      addonRight={<Text size="sm">{text}</Text>}
      defaultValue={url}
    />
  </div>
)

const TopContent = ({
  icon,
  heading,
}: {
  icon?: ReactNode
  heading: string | ReactNode
}) => (
  <div className="col-span-full">
    <div className="flex justify-between">
      <div className="flex items-center space-x-3">
        {icon && <Icon icon={icon} variant="ghost" size="xl" />}
        {typeof heading === 'string' && <H2>{heading}</H2>}
        {typeof heading === 'object' && heading}
      </div>
      <QuickNav />
    </div>
  </div>
)

function SearchInput() {
  return (
    <Input
      variant="outline"
      addonLeft={<FontAwesomeIcon icon={faSearch} />}
      addonUnstyled
      placeholder="Search"
    />
  )
}

function Tip({ position, children }) {
  return (
    <Tooltip
      position={position}
      html={<div className="px-2 py-0.5">{children}</div>}
    >
      <FontAwesomeIcon icon={faInfoCircle} className="text-sm" />
    </Tooltip>
  )
}

function Dot({ color = 'gray-400' }) {
  return (
    <div className="w-10 h-10 flex-shrink-0 flex items-center justify-center">
      <div className={`rounded-full bg-${color} w-1.5 h-1.5`} />
    </div>
  )
}

function NameAndBadge({ name, role, onClick }: User) {
  return (
    <div className="flex space-x-2 items-center">
      {onClick ? (
        <button onClick={onClick}>
          <Text size="sm" className="cursor-pointer" underline>
            {name}
          </Text>
        </button>
      ) : (
        <Text size="sm">{name}</Text>
      )}
      {role && (
        <Badge variant="primary" type="subtle">
          {role}
        </Badge>
      )}
    </div>
  )
}

function Tags({ tags }: { tags: Array<string> }) {
  if (tags?.length < 1) return null

  return (
    <div className="flex flex-wrap -mt-2">
      {tags.map((tag) => (
        <Tag
          tagType="round"
          variant="primary"
          className="mr-2 mt-2"
          size="sm"
          key={tag}
        >
          {tag}
        </Tag>
      ))}
    </div>
  )
}

function Footer({ className }) {
  return (
    <footer className={cn('flex items-center space-x-2 py-10', className)}>
      <Text size="xs">{commonStrings.empowered}</Text>
      <img src="/assets/small-om-logo.png" className="h-6" />
    </footer>
  )
}

function ButtonGroup({ children }) {
  return <div className="space-x-4 pt-6">{children}</div>
}

export {
  ButtonGroup,
  Dot,
  Footer,
  NameAndBadge,
  Optional,
  InputCopyToClipboard,
  SearchInput,
  Tags,
  TopContent,
  Tip as Tooltip,
}
