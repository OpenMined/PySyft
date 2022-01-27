import { Text } from '@/omui'
import type { ReactNode } from 'react'

export interface CardItemProps {
  text: string
  value: number | string
  TextComponent?: ReactNode
  ValueComponent?: ReactNode
}

function CardItem({ text, value, TextComponent = CardItemText, ValueComponent = CardItemValue }) {
  return (
    <div className="flex space-x-2 items-center">
      <TextComponent>{text}:</TextComponent>
      <ValueComponent>{value}</ValueComponent>
    </div>
  )
}

function CardItemText(props) {
  return <Text as="p" bold size="sm" {...props} />
}

function CardItemValue(props) {
  return <Text size="sm" {...props} />
}

export { CardItem }
