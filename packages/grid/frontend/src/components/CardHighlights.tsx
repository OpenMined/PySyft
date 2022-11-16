import { Divider, Text } from '@/omui'
import type { ReactNode } from 'react'

interface HighlightItem {
  text: string
  value: number | string | ReactNode
}

function HighlightedItem({ text = '', value = 0 }: HighlightItem) {
  return (
    <div className="py-4">
      <Text as="p" size="xl">
        {value}
      </Text>
      <Text as="p" size="sm">
        {text}
      </Text>
    </div>
  )
}

interface HighlightsProps {
  highlights: Array<HighlightItem>
}

export function Highlights({ highlights = [] }: HighlightsProps) {
  return (
    <div className="flex space-x-6">
      {highlights.map((highlight, index) => (
        <>
          <HighlightedItem text={highlight.text} value={highlight.value} />
          {/* TODO: could be extracted if we ever find different use cases... */}
          {index + 1 !== highlights.length && (
            <div className="self-stretch py-4">
              <Divider orientation="vertical" color="light" />
            </div>
          )}
        </>
      ))}
    </div>
  )
}
