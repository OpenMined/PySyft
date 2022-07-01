import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import cases from 'jest-in-case'
import { axe } from 'jest-axe'

import { Select } from '../Select'

const mockOptions = [
  { label: 'Foundations of Private Computation', value: 'foundations' },
  { label: 'Our Privacy Opportunity', value: 'privacy-opportunity' },
  {
    label: 'Federated Learning Across Enterprises',
    value: 'federated-learning',
  },
]

cases(
  'styles:classes',
  ({ params, result }) => {
    render(<Select data-testid="test-id" {...params} />)

    const selectElement = screen.getByTestId('test-id')
    expect(selectElement).toHaveClass(result)
  },
  [
    {
      name: 'default: default size and default states',
      params: { options: [] },
      result: 'border-gray-100 py-2 text-md',
    },
    {
      name: 'size: changing size and default states',
      params: { options: [], size: 'sm' },
      result: 'border-gray-100 py-1 text-sm',
    },
    {
      name: 'custom-state/error: default size and error state',
      params: { options: [], error: true },
      result: 'border-error-500 py-2 text-md',
    },
    {
      name: 'custom-state/disabled: default size and disabled state',
      params: { options: [], disabled: true },
      result: 'opacity-40 pointer-events-none py-2 text-md',
    },
  ]
)

describe('placeholder text', () => {
  test("display the placeholder prop as placeholder when the value isn't set", () => {
    const placeholder = "What You'll Learn"
    render(
      <Select
        data-testid="select-testid"
        placeholder={placeholder}
        options={mockOptions}
      />
    )
    const component = screen.getByTestId('select-testid')

    expect(component).toHaveTextContent(placeholder)
  })

  test('display the selected value as placeholder', () => {
    const { value, label } = mockOptions[2]
    render(
      <Select data-testid="select-testid" value={value} options={mockOptions} />
    )
    const component = screen.getByTestId('select-testid')

    expect(component).toHaveTextContent(label)
  })
})

describe('mouse navigation', () => {
  test('options are selectable via mouse clicks', () => {
    const mockFn = jest.fn()
    render(
      <Select
        data-testid="select-testid"
        onChange={mockFn}
        options={mockOptions}
      />
    )
    const component = screen.getByTestId('select-testid')

    // Open dropdown
    userEvent.click(component)

    const optionElement = screen.getByRole('menuitem', {
      name: mockOptions[0].label,
    })

    userEvent.click(optionElement)

    expect(mockFn).toHaveBeenCalledWith(mockOptions[0].value)
  })

  test('select is not focusable when disabled', () => {
    render(
      <Select data-testid="select-testid" disabled options={mockOptions} />
    )
    const component = screen.getByTestId('select-testid')

    userEvent.tab()
    expect(component).not.toHaveFocus()
  })
})

describe('keyboard navigation', () => {
  test('select is tabbable and options are selectable via keyboard pressing keys', () => {
    const mockFn = jest.fn()
    render(
      <Select
        data-testid="select-testid"
        onChange={mockFn}
        options={mockOptions}
      />
    )
    const component = screen.getByTestId('select-testid')

    // Tab to focus component
    userEvent.tab()
    expect(component).toHaveFocus()

    // Open dropdown
    userEvent.type(component, '{enter}', { skipClick: true })

    // Click arrow down to walk on options
    userEvent.type(component, '{arrowdown}', { skipClick: true })
    const firstOptionValue = mockOptions[0].value

    // Select option with space
    userEvent.type(component, '{space}', { skipClick: true })
    expect(mockFn).toHaveBeenCalledWith(firstOptionValue)
  })

  test('select is not focusable when disabled', () => {
    render(
      <Select data-testid="select-testid" disabled options={mockOptions} />
    )
    const component = screen.getByTestId('select-testid')

    userEvent.tab()
    expect(component).not.toHaveFocus()
  })
})

describe('accessibility', () => {
  test('set attributes for open state when clicking on select', async () => {
    const mockFn = jest.fn()
    const { container } = render(
      <Select
        data-testid="select-testid"
        onChange={mockFn}
        options={mockOptions}
      />
    )
    const component = screen.getByTestId('select-testid')
    expect(component).not.toHaveAttribute('aria-expanded')
    expect(component).not.toHaveAttribute('data-active')
    userEvent.tab()
    expect(component).toHaveFocus()
    expect(await axe(container)).toHaveNoViolations()

    // Open dropdown
    userEvent.type(component, '{enter}', { skipClick: true })
    expect(component).toHaveAttribute('aria-expanded', 'true')
    expect(component).toHaveAttribute('data-active', 'true')
    expect(await axe(container)).toHaveNoViolations()

    // Click arrow down to walk on options
    userEvent.type(component, '{arrowdown}{arrowdown}', { skipClick: true })
    const { value } = mockOptions[1]
    expect(component).toHaveAttribute(
      'aria-activedescendant',
      `omui-menu-item-${value}`
    )
    expect(await axe(container)).toHaveNoViolations()
  })
})
