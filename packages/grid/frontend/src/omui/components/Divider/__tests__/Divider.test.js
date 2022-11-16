import { render, screen } from '@testing-library/react'
import cases from 'jest-in-case'
import { axe } from 'jest-axe'

import { Divider } from '../Divider'

describe('Divider', () => {
  describe('accessibility', () => {
    test('No accessibility violation caught', async () => {
      const { container } = render(<Divider />)
      const result = await axe(container)
      expect(result).toHaveNoViolations()
    })

    cases(
      'accesibility:aria-orientation',
      ({ params, result }) => {
        render(<Divider data-testid="test-id" {...params} />)
        const dividerElement = screen.getByTestId('test-id')
        expect(dividerElement).toHaveAttribute('aria-orientation', result)
      },
      [
        {
          name: 'Default props, divider aria-orientation is horizontal',
          params: {},
          result: 'horizontal',
        },
        {
          name: 'Prop orientation="horizontal", then aria-orientation=horizontal',
          params: { orientation: 'horizontal' },
          result: 'horizontal',
        },
        {
          name: 'Prop orientation="vertical", then aria-orientation=vertical',
          params: { orientation: 'vertical' },
          result: 'vertical',
        },
      ]
    )

    cases(
      'accesibility:role',
      ({ params }) => {
        render(<Divider data-testid="test-id" {...params} />)
        const dividerElement = screen.getByTestId('test-id')
        expect(dividerElement).toHaveAttribute('role', 'separator')
      },
      [
        {
          name: 'Default props',
          params: {},
        },
        {
          name: 'Prop orientation="horizontal"',
          params: { orientation: 'horizontal' },
        },
        {
          name: 'Prop orientation="vertical"',
          params: { orientation: 'vertical' },
        },
      ]
    )
  })

  describe('styles:classes', () => {
    cases(
      'styles:classes:color',
      ({ params, result }) => {
        render(<Divider data-testid="test-id" {...params} />)
        const dividerElement = screen.getByTestId('test-id')
        expect(dividerElement).toHaveClass(result)
      },
      [
        {
          name: 'Default props',
          params: {},
          result: 'border-gray-200 dark:border-gray-700',
        },
        {
          name: 'Prop color="dynamic"',
          params: { color: 'dynamic' },
          result: 'border-gray-200 dark:border-gray-700',
        },
        {
          name: 'Prop color="black"',
          params: { color: 'black' },
          result: 'border-gray-900',
        },
      ]
    )
  })
})
