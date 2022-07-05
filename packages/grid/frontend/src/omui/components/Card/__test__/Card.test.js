import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import cases from 'jest-in-case'

import { Card } from '../Card'

cases(
  'Validate classes by variant',
  ({ params, result }) => {
    render(
      <Card data-testid="test-id" {...params}>
        Omui
      </Card>
    )

    const cardElement = screen.getByTestId('test-id')
    expect(cardElement).toHaveClass(result)
  },
  [
    {
      name: 'Should be default with M size and default variant',
      params: {},
      result: 'px-10 py-6 max-w-md bg-gray-800 hover:from-black',
    },
    {
      name: 'Should have S Card size',
      params: { size: 'S' },
      result: 'p-6 max-w-sm',
    },
    {
      name: 'Should have disabled classes',
      params: { disabled: true },
      result: 'opacity-40 pointer-events-none',
    },
    {
      name: 'Should have Coming variant classes',
      params: { variant: 'coming' },
      result: 'bg-gray-100 hover:bg-gray-50',
    },
    {
      name: 'Should have Progress variant classes',
      params: { variant: 'progress' },
      result: 'bg-gray-50 hover:from-gradient-white',
    },
    {
      name: 'Should have Completed variant classes',
      params: { variant: 'completed' },
      result: 'bg-gray-50 hover:bg-gray-50',
    },
  ]
)

describe('Render the Progress Card correctly', () => {
  test('Render the right footer with progress', () => {
    render(<Card data-testid="test-id" variant="progress" progress={90} />)

    const cardFooter = screen.getByRole('contentinfo')
    const progressDiv = cardFooter.getElementsByClassName(
      'absolute h-1 to-primary-500'
    )[0]
    expect(progressDiv).toHaveStyle('width: 90%')
  })

  test('Render the progress with a ranged value between 0 and 100', () => {
    render(<Card data-testid="test-id" variant="progress" progress={120} />)

    const cardFooter = screen.getByRole('contentinfo')
    const progressDiv = cardFooter.getElementsByClassName(
      'absolute h-1 to-primary-500'
    )[0]
    expect(progressDiv).toHaveStyle('width: 100%')
  })

  test('Render the progress formatting value with min = 0', () => {
    render(<Card data-testid="test-id" variant="progress" progress={-40} />)

    const cardFooter = screen.getByRole('contentinfo')
    const progressDiv = cardFooter.getElementsByClassName(
      'absolute h-1 to-primary-500'
    )[0]
    expect(progressDiv).toHaveStyle('width: 0%')
  })

  test('Render the progress formatting value with max = 100', () => {
    render(<Card data-testid="test-id" variant="progress" progress={-40} />)

    const cardFooter = screen.getByRole('contentinfo')
    const progressDiv = cardFooter.getElementsByClassName(
      'absolute h-1 to-primary-500'
    )[0]
    expect(progressDiv).toHaveStyle('width: 0%')
  })

  test('Render the progress percentage value', () => {
    const randomValue = Math.ceil(Math.random() * 100)
    render(
      <Card data-testid="test-id" variant="progress" progress={randomValue} />
    )

    const percentageText = screen.getByText(`${randomValue}%`)
    expect(percentageText).toBeInTheDocument()
  })
})

describe('Render the Tags Card correctly', () => {
  test('Render the right footer with tags list', () => {
    render(
      <Card
        data-testid="test-id"
        tags={[{ name: 'First tag' }, { name: 'Second tag' }]}
      />
    )

    const cardList = screen.getByRole('list')
    expect(cardList).toBeInTheDocument()
  })

  test('Render the tags list with clickable elements', () => {
    const mockFn = jest.fn()
    render(
      <Card
        data-testid="test-id"
        tags={[{ name: 'First tag', onClick: mockFn }, { name: 'Second tag' }]}
      />
    )

    const firstTagElement = screen.getByText(/First tag/)
    userEvent.click(firstTagElement)
    expect(mockFn).toHaveBeenCalledTimes(1)
  })
})
