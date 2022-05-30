import { render, screen } from '@testing-library/react'
import cases from 'jest-in-case'

import { Badge } from '../Badge'
import { Text } from '../../Typography/Text'

describe('Badge', () => {
  cases(
    'styles:classes',
    ({ params, result }) => {
      render(
        <Badge data-testid="test-id" {...params}>
          OMui
        </Badge>
      )

      const textElement = screen.getByTestId('test-id')
      expect(textElement).toHaveClass(result)
    },
    [
      {
        name: 'Should have colors based on default variant: primary',
        params: {},
        result:
          'border-primary-500 text-primary-600 dark:border-primary-200 dark:text-primary-200',
      },
      {
        name: 'Should have colors based on variant gray',
        params: { variant: 'gray' },
        result:
          'border-gray-500 text-gray-600 dark:border-gray-200 dark:text-gray-200',
      },
      {
        name: 'Should have colors based on variant primary',
        params: { variant: 'primary' },
        result:
          'border-primary-500 text-primary-600 dark:border-primary-200 dark:text-primary-200',
      },
      {
        name: 'Should have colors based on variant danger',
        params: { variant: 'danger' },
        result:
          'border-error-500 text-error-600 dark:border-error-200 dark:text-error-200',
      },
      {
        name: 'Should have classes based on default type: outline',
        params: {},
        result:
          'text-primary-600 dark:border-primary-200 dark:text-primary-200',
      },
      {
        name: 'Should have classes based on type subtle',
        params: { type: 'subtle' },
        result: 'bg-primary-100 text-primary-600',
      },
      {
        name: 'Should have classes based on type solid',
        params: { type: 'solid' },
        result: 'bg-primary-500 text-white',
      },
      {
        name: 'Should have classes based on type solid following gray custom rule',
        params: { type: 'solid', variant: 'gray' },
        result: 'bg-gray-800 text-primary-200',
      },
    ]
  )

  describe('styles:typography', () => {
    test('Badge text is text-xs', () => {
      render(<Badge>OMui</Badge>)

      const textElement = screen.getByText(/omui/i)
      expect(textElement).toHaveClass('text-xs')
    })

    test('it is possible to override text size by using a Text component', () => {
      render(
        <Badge data-testid="test-id">
          <Text size="lg">OMui LG</Text>
        </Badge>
      )

      const outerTextElement = screen
        .getByTestId('test-id')
        .querySelector('span')
      expect(outerTextElement).toHaveClass('text-xs')
      const insideText = screen.getByText(/omui/i)
      expect(insideText).toHaveClass('text-lg')
    })
  })
})
