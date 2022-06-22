import React from 'react'
import { render, screen } from '@testing-library/react'
import { axe } from 'jest-axe'

import { Sidebar } from '../Sidebar'

describe('Sidebars', () => {
  describe('sidebar:header', () => {
    test('renders sidebar with header with horizontal divider', () => {
      render(
        <Sidebar data-testid="list-id" header="Title">
          Children :)
        </Sidebar>
      )

      const titleElement = screen.queryByText('Title')
      expect(titleElement).toBeDefined()

      const dividerElement = screen.getByRole('separator')
      expect(dividerElement).toHaveAttribute('aria-orientation', 'horizontal')
    })

    test('contains no axe violations', async () => {
      const { container } = render(
        <Sidebar data-testid="list-id" header="Title">
          Children :)
        </Sidebar>
      )

      expect(await axe(container)).toHaveNoViolations()
    })
  })

  describe('sidebar:header', () => {
    test('renders the sidebar without header and vertical divider', () => {
      render(<Sidebar data-testid="list-id">Children :)</Sidebar>)

      const titleElement = screen.queryByText('Title')
      expect(titleElement).not.toBeInTheDocument()

      const dividerElement = screen.getByRole('separator')
      expect(dividerElement).toHaveAttribute('aria-orientation', 'vertical')
    })

    test('contains no axe violations', async () => {
      const { container } = render(
        <Sidebar data-testid="list-id">Children :)</Sidebar>
      )

      expect(await axe(container)).toHaveNoViolations()
    })
  })
})
