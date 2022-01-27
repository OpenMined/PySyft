import React from 'react'
import { render, screen } from '@testing-library/react'
import { axe } from 'jest-axe'

import {
  List,
  ListItem,
  UnorderedList,
  UnorderedListItem,
  OrderedList,
  OrderedListItem,
  ListAvatarItem,
  ListProgressItem,
  ListIconItem,
  ListItemContent,
} from '../List'

const list = Array.from(Array(10).keys()).map(i => `Item ${i}`)

describe('Lists', () => {
  describe('lists:default', () => {
    test('renders the list item as children', () => {
      render(
        <List data-testid="list-id">
          {list.map(i => (
            <ListItem key={i}>{i}</ListItem>
          ))}
        </List>
      )

      const listElement = screen.getByTestId('list-id')
      expect(listElement.childElementCount).toBe(10)
    })

    test('accepts sizes=[md,lg,xl,2xl,3xl]', async () => {
      render(
        <List data-testid="list-id" size="3xl">
          {list.map(i => (
            <ListItem key={i}>{i}</ListItem>
          ))}
        </List>
      )

      const listElement = screen.getByTestId('list-id')
      const text = screen.getByText(/item 1/i)
      expect(listElement).toHaveClass('space-y-2')
      expect(text).toHaveClass('font-rubik text-2xl')
    })

    test('contains no axe violations', async () => {
      const { container } = render(
        <List data-testid="list-id">
          {list.map(i => (
            <ListItem key={i}>{i}</ListItem>
          ))}
        </List>
      )

      const results = await axe(container)
      expect(results).toHaveNoViolations()
    })
  })

  describe('lists:unordered', () => {
    test('renders the list as list-disc', () => {
      render(
        <UnorderedList data-testid="list-id">
          {list.map(i => (
            <UnorderedListItem key={i}>{i}</UnorderedListItem>
          ))}
        </UnorderedList>
      )

      const listElement = screen.getByTestId('list-id')
      expect(listElement).toHaveClass('list-disc')
      expect(screen.getByText(/item 9/i)).toBeTruthy()
    })

    test('contains no axe violations', async () => {
      const { container } = render(
        <UnorderedList data-testid="list-id">
          {list.map(i => (
            <UnorderedListItem key={i}>{i}</UnorderedListItem>
          ))}
        </UnorderedList>
      )

      const results = await axe(container)
      expect(results).toHaveNoViolations()
    })
  })

  describe('lists:ordered', () => {
    test('renders the list as list-decimal', () => {
      render(
        <OrderedList data-testid="list-id">
          {list.map(i => (
            <OrderedListItem key={i}>{i}</OrderedListItem>
          ))}
        </OrderedList>
      )

      const listElement = screen.getByTestId('list-id')
      expect(listElement).toHaveClass('list-decimal')
      expect(screen.getByText(/item 9/i)).toBeTruthy()
    })

    test('contains no axe violations', async () => {
      const { container } = render(
        <OrderedList data-testid="list-id">
          {list.map(i => (
            <OrderedListItem key={i}>{i}</OrderedListItem>
          ))}
        </OrderedList>
      )

      const results = await axe(container)
      expect(results).toHaveNoViolations()
    })
  })

  describe('list:avatar', () => {
    test('renders the list of avatars and labels', () => {
      render(
        <List data-testid="list-id">
          {list.map(i => (
            <ListAvatarItem src={i} alt={i} key={i}>
              Title {i}
            </ListAvatarItem>
          ))}
        </List>
      )

      const avatarElements = screen.getAllByRole('img')
      const textElements = screen.getAllByText(/Title Item/)
      expect(avatarElements.length).toBe(list.length)
      expect(textElements.length).toBe(list.length)
    })

    test('renders the list with ListItemContent', () => {
      render(
        <List data-testid="list-id">
          {list.map(i => (
            <ListAvatarItem src={i} alt={i} key={i}>
              <ListItemContent label={`Title ${i}`} description={`Desc ${i}`} />
            </ListAvatarItem>
          ))}
        </List>
      )

      const labelElements = screen.getAllByText(/Title/)
      const descElements = screen.getAllByText(/Desc/)
      expect(labelElements.length).toBe(list.length)
      expect(descElements.length).toBe(list.length)
    })

    test('avatar receives size from the list', () => {
      render(
        <List data-testid="list-id" size="3xl">
          {list.map(i => (
            <ListAvatarItem src={i} alt={i} key={i}>
              <ListItemContent label={`Title ${i}`} description={`Desc ${i}`} />
            </ListAvatarItem>
          ))}
        </List>
      )

      const firstAvatarElement = screen.getByAltText('Item 0')
      const firstLabelElement = screen.getByText('Title Item 0')
      const firstDescElement = screen.getByText('Desc Item 0')

      expect(firstAvatarElement).toHaveClass('w-20 h-20')
      expect(firstLabelElement).toHaveClass('text-2xl')
      expect(firstDescElement).toHaveClass('text-lg')
    })

    test('no axe violations with a string', async () => {
      const { container } = render(
        <List data-testid="list-id">
          {list.map(i => (
            <ListAvatarItem src={i} alt="avatar" key={i}>
              {i}
            </ListAvatarItem>
          ))}
        </List>
      )

      const results = await axe(container)
      expect(results).toHaveNoViolations()
    })

    test('no axe violations with a ListItemContent', async () => {
      const { container } = render(
        <List data-testid="list-id">
          {list.map(i => (
            <ListAvatarItem src={i} alt="avatar" key={i}>
              <ListItemContent label={i} description={i} />
            </ListAvatarItem>
          ))}
        </List>
      )

      const results = await axe(container)
      expect(results).toHaveNoViolations()
    })
  })

  describe('lists:progress', () => {
    test('renders the progress icon and children', () => {
      render(
        <List data-testid="list-id">
          {list.map(i => (
            <ListProgressItem key={i}>{i}</ListProgressItem>
          ))}
        </List>
      )

      const textElements = screen.getAllByText(/Item/)
      expect(textElements.length).toBe(list.length)
    })

    test('default progress list size follows the parent list size', () => {
      render(
        <List data-testid="list-id">
          {list.map(i => (
            <ListProgressItem key={i}>{i}</ListProgressItem>
          ))}
        </List>
      )

      const firstIconElement = screen.getByTestId('list-id').querySelector('li')
      const iconBox = firstIconElement.firstChild
      expect(iconBox).toHaveClass('w-10 w-10')
      const iconElement = iconBox.firstChild
      expect(iconElement).toHaveClass('w-6 h-6')
      const iconSvg = iconElement.firstChild
      expect(iconSvg).toHaveClass('w-3 h-3')
    })

    test('changing the list size triggers a change of the progress icon size', () => {
      render(
        <List data-testid="list-id" size="2xl">
          {list.map(i => (
            <ListProgressItem key={i}>{i}</ListProgressItem>
          ))}
        </List>
      )

      const firstIconElement = screen.getByTestId('list-id').querySelector('li')
      const iconBox = firstIconElement.firstChild
      expect(iconBox).toHaveClass('w-20 w-20')
      const iconElement = iconBox.firstChild
      expect(iconElement).toHaveClass('w-11 h-11')
      const iconSvg = iconElement.firstChild
      expect(iconSvg).toHaveClass('w-4.5 h-4.5')
    })

    test('contains no axe violations', async () => {
      const { container } = render(
        <List data-testid="list-id">
          {list.map(i => (
            <ListProgressItem key={i}>{i}</ListProgressItem>
          ))}
        </List>
      )

      const results = await axe(container)
      expect(results).toHaveNoViolations()
    })
  })

  const RandomIcon = ({ className }) => (
    <svg className={className} role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 496 512">
      <path
        fill="currentColor"
        d="M248 8C111 8 0 119 0 256s111 248 248 248 248-111 248-248S385 8 248 8zm80 168c17.7 0 32 14.3 32 32s-14.3 32-32 32-32-14.3-32-32 14.3-32 32-32zm-160 0c17.7 0 32 14.3 32 32s-14.3 32-32 32-32-14.3-32-32 14.3-32 32-32zm194.8 170.2C334.3 380.4 292.5 400 248 400s-86.3-19.6-114.8-53.8c-13.6-16.3 11-36.7 24.6-20.5 22.4 26.9 55.2 42.2 90.2 42.2s67.8-15.4 90.2-42.2c13.4-16.2 38.1 4.2 24.6 20.5z"
      />
    </svg>
  )

  describe('lists:icons', () => {
    // TODO: Expand this test to all list types
    test('renders all list elements', () => {
      render(
        <List data-testid="list-id">
          {list.map(i => (
            <ListIconItem icon={RandomIcon} key={i}>
              {i}
            </ListIconItem>
          ))}
        </List>
      )

      const textElements = screen.getAllByText(/Item/)
      expect(textElements.length).toBe(list.length)
    })

    test('icon size is determined by the list size', () => {
      // TODO: Transform with cases
      const { rerender } = render(
        <List data-testid="list-id">
          {list.map(i => (
            <ListIconItem icon={RandomIcon} key={i}>
              {i}
            </ListIconItem>
          ))}
        </List>
      )

      const firstListElement = screen.getByTestId('list-id').querySelector('li')
      const iconBox = firstListElement.firstChild
      expect(iconBox).toHaveClass('w-10 h-10')
      const iconSvg = iconBox.firstChild.firstChild
      expect(iconSvg).toHaveClass('w-4 h-4')

      rerender(
        <List data-testid="list-id" size="2xl">
          {list.map(i => (
            <ListIconItem icon={RandomIcon} key={i}>
              {i}
            </ListIconItem>
          ))}
        </List>
      )

      const newFirstListElement = screen.getByTestId('list-id').querySelector('li')
      const newIconBox = newFirstListElement.firstChild
      expect(newIconBox).toHaveClass('w-20 h-20')
      const newIconSvg = newIconBox.firstChild.firstChild
      expect(newIconSvg).toHaveClass('w-7 h-7')
    })

    test('contains no axe violations', async () => {
      // TODO: they should be instantiated using cases, perhaps?
      const { container } = render(
        <List data-testid="list-id">
          {list.map(i => (
            <ListProgressItem key={i}>{i}</ListProgressItem>
          ))}
        </List>
      )

      const results = await axe(container)
      expect(results).toHaveNoViolations()
    })
  })
})
