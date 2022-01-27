import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { axe } from 'jest-axe'
import cases from 'jest-in-case'

import { Button, IconButton } from '../Button'

describe('Button', () => {
  cases(
    'styles:classes',
    ({ params, result }) => {
      render(<Button data-testid="test-id" {...params} />)

      const element = screen.getByTestId('test-id')
      expect(element).toHaveClass(result)
    },
    [
      {
        name: 'variants: variant gray',
        params: { variant: 'gray' },
        result:
          'bg-gray-800 text-primary-200 hover:from-gradient-white bg-gradient-to-r dark:hover:from-gradient-mostly-black',
      },
      {
        name: 'variants: variant primary',
        params: { variant: 'primary' },
        result: 'bg-primary-500 text-white hover:from-gradient-white bg-gradient-to-r',
      },
      {
        name: 'variants: variant outline',
        params: { variant: 'outline' },
        result:
          'bg-transparent text-primary-600 border-2 border-primary-500 hover:bg-primary-500 hover:text-white dark:text-primary-200 dark:border-primary-200 dark:hover:border-primary-500 dark:hover:text-white',
      },
      {
        name: 'variants: variant ghost',
        params: { variant: 'ghost' },
        result:
          'text-primary-600 dark:text-primary-200 hover:bg-primary-100 dark:hover:text-primary-600',
      },
      {
        name: 'variants: variant link',
        params: { variant: 'link' },
        result: 'text-primary-600 dark:text-primary-200 hover:underline',
      },
      {
        name: 'sizes: size lg',
        params: { size: 'lg' },
        result: 'px-4 py-3',
      },
      {
        name: 'sizes: size md',
        params: { size: 'md' },
        result: 'px-3 py-2',
      },
      {
        name: 'sizes: size sm',
        params: { size: 'sm' },
        result: 'p-2',
      },
      {
        name: 'sizes: size xs',
        params: { size: 'xs' },
        result: 'p-2',
      },
      {
        name: 'states: disabled',
        params: { disabled: true },
        result: 'opacity-40 pointer-events-none',
      },
    ]
  )

  cases(
    'render:icons',
    ({ params, testId = [] }) => {
      render(<Button {...params} />)

      testId.forEach(id => {
        const element = screen.getByTestId(id)
        expect(element).toBeInTheDocument()
      })
    },
    [
      {
        name: 'icons: left',
        params: { leftIcon: () => <i data-testid="left-icon" /> },
        testId: ['left-icon'],
      },
      {
        name: 'icons: right',
        params: { rightIcon: () => <i data-testid="right-icon" /> },
        testId: ['right-icon'],
      },
      {
        name: 'icons: both sides',
        params: {
          leftIcon: () => <i data-testid="left-icon" />,
          rightIcon: () => <i data-testid="right-icon" />,
        },
        testId: ['left-icon', 'right-icon'],
      },
    ]
  )

  describe('states', () => {
    test('button is not clickable when disabled', () => {
      const mockFn = jest.fn()
      render(
        <Button data-testid="test-id" disabled onClick={mockFn}>
          Omui
        </Button>
      )
      const buttonElement = screen.getByTestId('test-id')

      userEvent.click(buttonElement)
      expect(mockFn).not.toHaveBeenCalledTimes(1)
    })
  })

  describe('accessibility', () => {
    test('component do not have axe violations', async () => {
      const { container } = render(<Button>Omui</Button>)

      expect(await axe(container)).toHaveNoViolations()
    })
  })
})

const IconMock = props => (
  <svg {...props}>
    <path
      fill="currentColor"
      d="M248 8C111 8 0 119 0 256s111 248 248 248 248-111 248-248S385 8 248 8zm80 168c17.7 0 32 14.3 32 32s-14.3 32-32 32-32-14.3-32-32 14.3-32 32-32zm-160 0c17.7 0 32 14.3 32 32s-14.3 32-32 32-32-14.3-32-32 14.3-32 32-32zm194.8 170.2C334.3 380.4 292.5 400 248 400s-86.3-19.6-114.8-53.8c-13.6-16.3 11-36.7 24.6-20.5 22.4 26.9 55.2 42.2 90.2 42.2s67.8-15.4 90.2-42.2c13.4-16.2 38.1 4.2 24.6 20.5z"
    />
  </svg>
)

describe('Icon Button', () => {
  cases(
    'styles:classes',
    ({ params, result }) => {
      render(<IconButton data-testid="test-id" icon={IconMock} {...params} />)

      const element = screen.getByTestId('test-id')
      expect(element).toHaveClass(result)
    },
    [
      {
        name: 'variants: variant gray',
        params: { variant: 'gray' },
        result:
          'bg-gray-800 text-primary-200 hover:from-gradient-white bg-gradient-to-r dark:hover:from-gradient-mostly-black',
      },
      {
        name: 'variants: variant primary',
        params: { variant: 'primary' },
        result: 'bg-primary-500 text-white hover:from-gradient-white bg-gradient-to-r',
      },
      {
        name: 'variants: variant outline',
        params: { variant: 'outline' },
        result:
          'bg-transparent text-primary-600 border-2 border-primary-500 hover:bg-primary-500 hover:text-white dark:text-primary-200 dark:border-primary-200 dark:hover:border-primary-500 dark:hover:text-white',
      },
      {
        name: 'variants: variant ghost',
        params: { variant: 'ghost' },
        result:
          'text-primary-600 dark:text-primary-200 hover:bg-primary-100 dark:hover:text-primary-600',
      },
      {
        name: 'variants: variant link',
        params: { variant: 'link' },
        result: 'text-primary-600 dark:text-primary-200 hover:underline',
      },
      {
        name: 'sizes: size lg',
        params: { size: 'lg' },
        result: 'p-3.5',
      },
      {
        name: 'sizes: size md',
        params: { size: 'md' },
        result: 'p-3',
      },
      {
        name: 'sizes: size sm',
        params: { size: 'sm' },
        result: 'p-2.5',
      },
      {
        name: 'sizes: size xs',
        params: { size: 'xs' },
        result: 'p-2',
      },
      {
        name: 'states: disabled',
        params: { disabled: true },
        result: 'opacity-40 pointer-events-none',
      },
    ]
  )

  describe('states', () => {
    test('button is not clickable when disabled', () => {
      const mockFn = jest.fn()
      render(<IconButton data-testid="test-id" disabled onClick={mockFn} icon={IconMock} />)
      const buttonElement = screen.getByTestId('test-id')

      userEvent.click(buttonElement)
      expect(mockFn).not.toHaveBeenCalledTimes(1)
    })
  })

  describe('accessibility', () => {
    test('component have axe violations when no label is passed', async () => {
      const { container } = render(<IconButton icon={IconMock} />)

      expect(await axe(container)).not.toHaveNoViolations()
    })

    test('component have no axe violations when aria- label is passed', async () => {
      const { container } = render(<IconButton aria-label="Close window" icon={IconMock} />)

      expect(await axe(container)).toHaveNoViolations()
    })
  })
})
