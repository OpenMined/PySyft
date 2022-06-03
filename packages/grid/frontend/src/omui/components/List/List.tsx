import React, { createContext, useContext, useMemo } from 'react'
import cn from 'classnames'
import { Avatar } from '../Avatar/Avatar'
import { Text } from '../Typography/Text'
import { Icon } from '../Icon/Icon'
import { Progress } from '../../icons'
import type {
  ReactNode,
  ElementType,
  HTMLAttributes,
  ComponentPropsWithoutRef,
  HTMLProps,
  PropsWithRef,
} from 'react'
import type { IconSizeProp } from '../Icon/Icon'
import type { TextSizeProp } from '../Typography/Text'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'

export type ListVariantProps =
  | 'bullet'
  | 'number'
  | 'avatar'
  | 'progress'
  | 'icon'
  | 'contained'
export type ListSizeProp = 'md' | 'lg' | 'xl' | '2xl' | '3xl'
export type ListVerticalPadding = 'py-0' | 'py-2'

interface Props extends HTMLAttributes<HTMLUListElement> {
  /**
   * The size of the elements of the list.
   * @defaultValue md
   */
  size?: ListSizeProp
  /**
   * The font size of the text.
   * @defaultValue md
   */
  component?: ElementType
}

export type ListProps = PropsWithRef<Props> & { horizontal?: boolean }

const ListContext = createContext<{ size: ListSizeProp }>({ size: 'md' })

const spaceBetweenList: ListByStringSizes<string> = {
  md: 'space-y-3',
  lg: 'space-y-2',
  xl: 'space-y-2',
  '2xl': 'space-y-2',
  '3xl': 'space-y-2',
}

type ListByStringSizes<T> = {
  [k in ListSizeProp]: T
}

type TextSizes = ListByStringSizes<TextSizeProp>
type ListPaddings = ListByStringSizes<ListVerticalPadding>

const textSize: TextSizes = {
  md: 'md',
  lg: 'md',
  xl: 'lg',
  '2xl': 'xl',
  '3xl': '2xl',
}

const verticalPadding: ListPaddings = {
  md: 'py-0',
  lg: 'py-2',
  xl: 'py-2',
  '2xl': 'py-2',
  '3xl': 'py-2',
}

function List({
  component: Component = 'ul',
  size = 'md',
  horizontal = false,
  className,
  children,
  ...props
}: ListProps) {
  const classes = cn(
    `text-gray-600 dark:text-gray-200`,
    horizontal ? 'space-y-0' : spaceBetweenList[size],
    verticalPadding[size],
    className
  )
  const currentSize = useMemo<ListSizeProp>(() => size, [size])
  return (
    <ListContext.Provider value={{ size: currentSize }}>
      <Component
        className={classes}
        /**
         * This role intends to fix the Safari accessibility issue with list-style-type: none
         * @see https://www.scottohara.me/blog/2019/01/12/lists-and-safari.html
         */
        role="list"
        {...props}
      >
        {children}
      </Component>
    </ListContext.Provider>
  )
}

const spaceInnerItems: ListByStringSizes<string> = {
  md: 'space-x-2',
  lg: 'space-x-2',
  xl: 'space-x-2',
  '2xl': 'space-x-3',
  '3xl': 'space-x-4',
}

function RenderListChildren({ children }: { children: ReactNode }) {
  const { size } = useContext(ListContext)
  return (
    <>
      {typeof children === 'string' ? (
        <Text size={textSize[size]}>{children}</Text>
      ) : (
        children
      )}
    </>
  )
}

function ListItem(props: ComponentPropsWithoutRef<'li'>) {
  const { size } = useContext(ListContext)
  const { children, className, ...rest } = props
  return (
    <li
      className={cn('flex items-center', spaceInnerItems[size], className)}
      {...rest}
    >
      <RenderListChildren>{children}</RenderListChildren>
    </li>
  )
}

function PrefixedItem(props: ComponentPropsWithoutRef<'div'>) {
  const { children, ...rest } = props
  return (
    <div className={cn('flex items-center')} {...rest}>
      <RenderListChildren>{children}</RenderListChildren>
    </div>
  )
}

const containerSize: ListByStringSizes<string> = {
  md: 'w-10 h-10',
  lg: 'w-14 h-14',
  xl: 'w-16 h-16',
  '2xl': 'w-20 h-20',
  '3xl': 'w-24 h-24',
}

function ListInnerContainer(props: HTMLProps<HTMLDivElement>) {
  const { size } = useContext(ListContext)
  return (
    <div
      className={cn('flex items-center justify-center', containerSize[size])}
      {...props}
    />
  )
}

const bulletSize: ListByStringSizes<string> = {
  md: 'w-1.5 h-1.5',
  lg: 'w-2 h-2',
  xl: 'w-2.5 h-2.5',
  '2xl': 'w-3 h-3',
  '3xl': 'w-3.5 h-3.5',
}

function Bullet({
  size,
  ...props
}: { size: string } & ComponentPropsWithoutRef<'div'>) {
  const classes = cn(
    'rounded-full bg-gray-800 dark:bg-gray-200',
    bulletSize[size],
    props.className
  )
  return <div {...props} className={classes} />
}

function PrefixedListMarker({
  isOrdered,
  listNumber,
}: {
  isOrdered: boolean
  listNumber: number
}) {
  const { size } = useContext(ListContext)
  return isOrdered ? (
    <Text size={textSize[size]}>{listNumber}.</Text>
  ) : (
    <Bullet size={size} />
  )
}

function BuildPrefixedList({
  component = 'ul',
  className,
  ...props
}: ListProps) {
  const { children, ...rest } = props
  const isOrdered = component === 'ol'
  const classes = cn(
    isOrdered ? 'list-decimal' : 'list-disc',
    'list-inside',
    className
  )
  return (
    <List className={classes} component="ol" {...rest}>
      {React.Children.map(children, (child, index) => {
        return (
          <ListItem key={`ol-${props.id}-${index}`}>
            <ListInnerContainer>
              <PrefixedListMarker
                isOrdered={component === 'ol'}
                listNumber={index + 1}
              />
            </ListInnerContainer>
            {child}
          </ListItem>
        )
      })}
    </List>
  )
}

function OrderedList({ ...props }: Exclude<ListProps, 'component'>) {
  return <BuildPrefixedList {...props} component="ol" />
}

function UnorderedList({ ...props }: Exclude<ListProps, 'component'>) {
  return <BuildPrefixedList {...props} component="ul" />
}

export type ListAvatarItemProps = HTMLProps<HTMLLIElement> &
  Pick<HTMLProps<HTMLImageElement>, 'src' | 'alt'>

function ListAvatarItem({ src, alt, children, ...props }: ListAvatarItemProps) {
  const { size } = useContext(ListContext)
  return (
    <ListItem {...props}>
      <ListInnerContainer>
        <Avatar src={src} alt={alt} size={size} />
      </ListInnerContainer>
      <RenderListChildren>{children}</RenderListChildren>
    </ListItem>
  )
}

export type ListItemContentProps = {
  label: string
  description: string
  className?: string | string[]
}

const textSizeForLabel: TextSizes = {
  md: 'sm',
  lg: 'md',
  xl: 'lg',
  '2xl': 'xl',
  '3xl': '2xl',
}

const textSizeForDescription: TextSizes = {
  md: 'sm',
  lg: 'md',
  xl: 'lg',
  '2xl': 'lg',
  '3xl': 'lg',
}

const spacingBetweenText: ListByStringSizes<string> = {
  md: '',
  lg: '',
  xl: '',
  '2xl': 'space-y-0.5',
  '3xl': 'space-y-2',
}

function ListItemContent({
  label,
  description,
  className,
}: ListItemContentProps) {
  const { size } = useContext(ListContext)
  const isLabelBold =
    size !== '2xl' && size !== '3xl' && description !== undefined
  return (
    <div
      className={cn(
        'flex flex-col w-full',
        spacingBetweenText[size],
        className
      )}
    >
      <Text size={textSizeForLabel[size]} bold={isLabelBold}>
        {label}
      </Text>
      <Text size={textSizeForDescription[size]}>{description}</Text>
    </div>
  )
}

const iconSize: ListByStringSizes<string> = {
  md: 'w-4 h-4',
  lg: 'w-4.5 h-4.5',
  xl: 'w-5 h-5',
  '2xl': 'h-7 w-7',
  '3xl': 'h-9 w-9',
}

const faIconSize: ListByStringSizes<string> = {
  md: 'text-md',
  lg: 'text-lg',
  xl: 'text-xl',
  '2xl': 'text-2xl',
  '3xl': 'text-3xl',
}

export type ListIconProps = HTMLProps<HTMLLIElement> & {
  icon: ElementType
  iconColor?: string
}

function ListIconItem({
  icon,
  children,
  iconColor = 'text-current',
  ...props
}: ListIconProps) {
  const { size } = useContext(ListContext)
  return (
    <ListItem {...props}>
      <ListInnerContainer>
        <Icon
          icon={icon}
          variant="ghost"
          className={cn(iconSize[size], iconColor)}
        />
      </ListInnerContainer>
      <RenderListChildren>{children}</RenderListChildren>
    </ListItem>
  )
}

function ListFAIconItem({
  icon,
  children,
  iconColor = 'text-current',
  ...props
}: ListIconProps) {
  const { size } = useContext(ListContext)
  return (
    <ListItem {...props}>
      <ListInnerContainer>
        <FontAwesomeIcon
          icon={icon}
          className={cn(iconColor, faIconSize[size], props.className)}
        />
      </ListInnerContainer>
      <RenderListChildren>{children}</RenderListChildren>
    </ListItem>
  )
}

const progressIconSize: ListByStringSizes<IconSizeProp> = {
  md: 'xs',
  lg: 'sm',
  xl: 'md',
  '2xl': 'lg',
  '3xl': 'xl',
}

function ListProgressItem({ children, ...props }: HTMLProps<HTMLLIElement>) {
  const { size } = useContext(ListContext)
  return (
    <ListItem {...props}>
      <ListInnerContainer>
        <Icon
          icon={Progress}
          variant="solid"
          size={progressIconSize[size]}
          className="text-gray-800 dark:text-white"
          containerProps={{ className: 'bg-success-500 dark:bg-success-400' }}
        />
      </ListInnerContainer>
      <RenderListChildren>{children}</RenderListChildren>
    </ListItem>
  )
}

export type ListContainedProps = HTMLProps<HTMLLIElement> & {
  containedValue: number | string
}

const containedTextSize: ListByStringSizes<TextSizeProp> = {
  md: 'xs',
  lg: 'sm',
  xl: 'md',
  '2xl': 'lg',
  '3xl': 'xl',
}

const containedCircleSize: ListByStringSizes<string> = {
  md: 'w-4.5 h-4.5',
  lg: 'w-6 h-6',
  xl: 'w-8 h-8',
  '2xl': 'w-10 h-10',
  '3xl': 'w-12 h-12',
}

export type ListContainedNumberProps = {
  value: string | number
}

function ListContainedNumber({ value }: ListContainedNumberProps) {
  const { size } = useContext(ListContext)
  const isBold = size !== '3xl'
  return (
    <div
      className={cn(
        'flex items-center justify-center rounded-full text-gray-50 bg-gray-800',
        containedCircleSize[size]
      )}
    >
      <Text bold={isBold} size={containedTextSize[size]}>
        {value}
      </Text>
    </div>
  )
}

function ListContainedItem({
  containedValue,
  children,
  ...props
}: ListContainedProps) {
  return (
    <ListItem {...props}>
      <ListInnerContainer>
        <ListContainedNumber value={containedValue} />
      </ListInnerContainer>
      <RenderListChildren>{children}</RenderListChildren>
    </ListItem>
  )
}

export {
  List,
  UnorderedList,
  PrefixedItem as UnorderedListItem,
  OrderedList,
  PrefixedItem as OrderedListItem,
  ListItem,
  ListInnerContainer,
  ListItemContent,
  ListAvatarItem,
  ListIconItem,
  ListFAIconItem,
  ListProgressItem,
  ListContainedItem,
  ListContainedNumber,
}
