import React, { forwardRef } from 'react'
import cn from 'classnames'
import { Text, TextSizeProp } from '../Typography/Text'
import { Image } from '../Image/Image'
import { Tag } from '../Tag/Tag'
import type { MouseEventHandler, HTMLAttributes, PropsWithRef } from 'react'

export type CardSizeProp = 'S' | 'M'
export type CardVariantProp = 'coming' | 'progress' | 'completed' | 'default'
export type CardTagProp = { name: string; onClick: () => void }

interface Props extends HTMLAttributes<HTMLButtonElement> {
  /**
   * The variant of the Card.
   * @defaultValue default
   */
  variant?: CardVariantProp
  /**
   * The size of the Card.
   * @defaultValue M
   */
  size?: CardSizeProp
  /**
   * The title of the Card.
   */
  title: string
  /**
   * The sub title of the Card.
   */
  subTitle: string
  /**
   * The image src attribute for the Card.
   */
  srcImage: string
  /**
   * The image alt attribute for the Card.
   */
  altImage: string
  /**
   * Used for displaying progress when card variant is `progress`.
   */
  progress?: number
  /**
   * Used for displaying tags when card variant is `coming` or `default`.
   */
  tags?: CardTagProp[]
  /**
   * Defines if the Card is disabled.
   */
  disabled?: boolean
  onClick?: MouseEventHandler<HTMLButtonElement>
}

export type CardProps = PropsWithRef<Props>

type Sizes = {
  [k in CardSizeProp]: TextSizeProp
}

type Variants = {
  [k in CardVariantProp]: string
}

const titleSizes: Sizes = {
  S: '2xl',
  M: '3xl',
}

const containerByVariant: Variants = {
  default: 'bg-gray-800 bg-gradient-to-r hover:from-black',
  coming: 'bg-gray-100 hover:bg-gray-50',
  progress: 'bg-gray-50 bg-gradient-to-l hover:from-gradient-white',
  completed: 'bg-gray-50 hover:bg-gray-50',
}

const tagTypeByVariant: Omit<Variants, 'completed' | 'progress'> = {
  default: 'primary',
  coming: 'default',
}

const titleByVariant: Variants = {
  default: 'text-gray-50',
  progress: 'text-gray-800',
  coming: 'text-gray-800',
  completed: 'text-gray-800',
}

const subTitleByVariant: Variants = {
  default: 'text-primary-200',
  coming: 'text-gray-600 group-hover:text-primary-600',
  progress: 'text-primary-600',
  completed: 'text-primary-600',
}

const defaultClass =
  'group flex flex-col shadow-card hover:shadow-card-hover transition transition-shadow transition-colors rounded-sm cursor-pointer text-left'

const Card = forwardRef<HTMLButtonElement, CardProps>(function Card(
  {
    title,
    subTitle,
    altImage,
    srcImage,
    onClick,
    variant = 'default',
    tags = [],
    progress,
    size = 'M',
    disabled,
    className,
    ...props
  },
  ref
) {
  const classes = cn(
    defaultClass,
    { 'opacity-40 pointer-events-none': disabled },
    { 'p-6 max-w-sm': size === 'S', 'px-10 py-6 max-w-md': size === 'M' },
    containerByVariant[variant],
    className
  )

  const titleClasses = cn(titleByVariant[variant])
  const subTitleClasses = cn(subTitleByVariant[variant])
  const imageClasses = cn('group-hover:opacity-50', {
    'my-4': size === 'S',
    'my-6': size === 'M',
  })

  const shouldRenderTags =
    (variant === 'default' || variant === 'coming') && tags.length > 0
  const shouldRenderProgress = variant === 'progress' && progress !== undefined
  const shouldRenderComplete = variant === 'completed'

  return (
    <button onClick={onClick} className={classes} {...props} ref={ref}>
      <Text className={subTitleClasses} mono>
        {subTitle}
      </Text>
      <Text as="h3" size={titleSizes[size]} className={titleClasses}>
        Foundations of Private Computation
      </Text>
      <Image
        className={imageClasses}
        ratio="1:1"
        src={srcImage}
        alt={altImage}
      />
      <footer>
        {shouldRenderTags && <TagsList tags={tags} variant={variant} />}
        {shouldRenderProgress && <Progress value={progress!} />}
        {shouldRenderComplete && <Complete />}
      </footer>
    </button>
  )
})

const TagsList = ({
  tags,
  variant,
}: {
  tags: CardTagProp[]
  variant: CardVariantProp
}) => {
  // WIP: Replace with Tag component
  return (
    <ul>
      {tags.map(({ name, onClick }) => (
        <li key={name}>
          <Tag variant={tagTypeByVariant[variant]} onClick={onClick}>
            {name}
          </Tag>
        </li>
      ))}
    </ul>
  )
}

const Progress = ({ value }: { value: number }) => {
  const rangedValue = Math.min(Math.max(value, 0), 100)
  return (
    <>
      <div className="flex justify-between items-center text-primary-600 mb-6">
        <Text mono>In Progress</Text>
        <Text mono>{rangedValue}%</Text>
      </div>
      <div className="bg-gray-200 h-1 rounded relative">
        <div
          style={{ width: `${rangedValue}%` }}
          className="absolute h-1 bg-gradient-to-r from-primary-200 to-primary-500 rounded"
        />
      </div>
    </>
  )
}

const Complete = () => (
  <div className="flex justify-between items-center text-primary-600">
    <Text mono>Today</Text>
    <Text mono>Complete!</Text>
  </div>
)

export { Card }
