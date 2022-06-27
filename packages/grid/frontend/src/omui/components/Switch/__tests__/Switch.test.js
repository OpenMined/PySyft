import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import cases from 'jest-in-case'
import { axe } from 'jest-axe'

import { Switch } from '../Switch'

cases(
  'styles:classes',
  ({ params, result }) => {
    render(<Switch data-testid="switch-testid" {...params} />)

    const element = screen.getByTestId('switch-testid')
    expect(element).toHaveClass(result)
  },
  [
    {
      name: 'default: gray bg, size md',
      params: {},
      result: 'bg-gray-200 w-9 h-5',
    },
    {
      name: 'size sm',
      params: { size: 'sm' },
      result: 'bg-gray-200 w-7 h-4',
    },
    {
      name: 'size lg',
      params: { size: 'lg' },
      result: 'bg-gray-200 w-13 h-7',
    },
    {
      name: 'checked state: primary bg',
      params: { checked: true, onChange: jest.fn },
      result: 'bg-primary-500 ',
    },
    {
      name: 'disabled: opacity 40%',
      params: { disabled: true },
      result: 'opacity-40',
    },
  ]
)

describe('mouse navigation', () => {
  test('toggle to true via mouse click', () => {
    const mockFn = jest.fn()
    render(
      <Switch data-testid="switch-testid" checked={false} onChange={mockFn} />
    )
    const component = screen.getByTestId('switch-testid')

    // Click on switch
    userEvent.click(component)

    expect(mockFn).toHaveBeenCalledWith(true)
  })

  test('toggle to false via mouse click', () => {
    const mockFn = jest.fn()
    render(<Switch data-testid="switch-testid" checked onChange={mockFn} />)
    const component = screen.getByTestId('switch-testid')

    // Click on switch
    userEvent.click(component)

    expect(mockFn).toHaveBeenCalledWith(false)
  })

  test('switch is not clickable when disabled', () => {
    const mockFn = jest.fn()
    render(<Switch data-testid="switch-testid" disabled onChange={mockFn} />)
    const component = screen.getByTestId('switch-testid')

    userEvent.click(component)
    expect(mockFn).not.toHaveBeenCalled()
  })
})

describe('keyboard navigation', () => {
  test('switch is tabbable and toggle works via keyboard', () => {
    const mockFn = jest.fn()
    render(<Switch data-testid="switch-testid" checked onChange={mockFn} />)
    const component = screen.getByTestId('switch-testid')

    // Tab to focus component
    userEvent.tab()
    expect(component).toHaveFocus()

    // Toggle value with enter
    userEvent.type(component, '{enter}', { skipClick: true })

    expect(mockFn).toHaveBeenCalledWith(false)
  })

  test('switch is tabbable and toggle works via keyboard : 2', () => {
    const mockFn = jest.fn()
    render(<Switch data-testid="switch-testid" checked onChange={mockFn} />)
    const component = screen.getByTestId('switch-testid')

    // Tab to focus component
    userEvent.tab()
    expect(component).toHaveFocus()

    // Toggle value with space
    userEvent.type(component, '{space}', { skipClick: true })

    expect(mockFn).toHaveBeenCalledWith(false)
  })

  test('switch is not focusable when disabled', () => {
    render(<Switch data-testid="switch-testid" disabled />)
    const component = screen.getByTestId('switch-testid')

    userEvent.tab()
    expect(component).not.toHaveFocus()
  })
})

describe('accessibility', () => {
  test("aria checked equals false when isn't checked", async () => {
    const { container } = render(<Switch />)
    const input = screen.getByRole('switch')
    expect(input).toHaveAttribute('aria-checked', 'false')
    // expect(await axe(container)).toHaveNoViolations()
  })

  test('aria checked equals true when is checked', async () => {
    const { container } = render(<Switch checked onChange={jest.fn} />)
    const input = screen.getByRole('switch')
    expect(input).toHaveAttribute('aria-checked', 'true')
    // expect(await axe(container)).toHaveNoViolations()
  })
})
