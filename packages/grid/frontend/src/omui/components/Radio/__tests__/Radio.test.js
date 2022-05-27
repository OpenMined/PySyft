import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import cases from 'jest-in-case'
import { axe } from 'jest-axe'

import { Radio } from '../Radio'
import { RadioGroup } from '../RadioGroup'

cases(
  'styles:classes',
  ({ params, result }) => {
    render(<Radio data-testid="radio-testid" {...params} />)

    const element = screen.getByTestId('radio-testid')
    expect(element).toHaveClass(result)
  },
  [
    {
      name: 'default: transparent bg and border gray',
      params: {},
      result: 'border-gray-400 bg-transparent',
    },
    {
      name: 'checked state: primary bg',
      params: { checked: true, onChange: jest.fn },
      result: 'bg-primary-500 border-primary-500',
    },
    {
      name: 'disabled: opacity 40%',
      params: { disabled: true },
      result: 'opacity-40 pointer-events-none',
    },
    {
      name: 'dark default: gray bg',
      params: {},
      result: 'border-gray-400 bg-transparent',
    },
    {
      name: 'dark checked state: light primary bg and custom bg image',
      params: { checked: true, onChange: jest.fn },
      result: 'dark:border-primary-400 dark:bg-primary-400 dark:bg-radio',
    },
    {
      name: 'disabled: opacity 40%',
      params: { disabled: true },
      result: 'opacity-40 pointer-events-none',
    },
  ]
)

describe('radio group', () => {
  test('radio group pass name to children Radio components', () => {
    const mockFn = jest.fn()
    render(
      <RadioGroup name="numbers" onChange={mockFn}>
        <Radio value="one" />
        <Radio value="two" />
        <Radio value="three" />
      </RadioGroup>
    )
    const components = screen.getAllByRole('radio')
    const everyNameIsNumbers = components.every(
      (i) => i.getAttribute('name') === 'numbers'
    )
    expect(everyNameIsNumbers).toBe(true)
  })

  test('only one value is returned from radio group onChange', () => {
    const mockFn = jest.fn()
    render(
      <RadioGroup onChange={mockFn}>
        <Radio value="one" />
        <Radio value="two" />
        <Radio data-testid="three-testid" value="three" />
      </RadioGroup>
    )
    const component = screen.getByTestId('three-testid')

    userEvent.click(component)
    expect(mockFn).toHaveBeenCalledWith('three')
  })

  test('onChange of a child Radio triggers to allow side effects', () => {
    const mockFn = jest.fn()
    const mockFn2 = jest.fn()
    render(
      <RadioGroup name="numbers" onChange={mockFn}>
        <Radio value="one" />
        <Radio data-testid="two-testid" value="two" onChange={mockFn2} />
        <Radio value="three" />
      </RadioGroup>
    )
    const component = screen.getByTestId('two-testid')

    userEvent.click(component)
    expect(mockFn).toHaveBeenCalledWith('two')
    expect(mockFn2).toHaveBeenCalledWith(
      expect.objectContaining({
        target: expect.objectContaining({ value: 'two' }),
      })
    )
  })
})

describe('mouse navigation', () => {
  test('toggle to true via mouse click', () => {
    const mockFn = jest.fn((e) => e.target.checked)
    render(
      <Radio data-testid="radio-testid" checked={false} onChange={mockFn} />
    )
    const component = screen.getByTestId('radio-testid')

    // Click on Radio
    userEvent.click(component)

    expect(mockFn).toHaveBeenCalledWith(
      expect.objectContaining({
        target: expect.objectContaining({ checked: false }),
      })
    )
  })

  test('not clickable when disabled', () => {
    const mockFn = jest.fn()
    render(<Radio data-testid="radio-testid" disabled onChange={mockFn} />)
    const component = screen.getByTestId('radio-testid')

    userEvent.click(component)
    expect(mockFn).not.toHaveBeenCalled()
  })
})

describe('keyboard navigation', () => {
  test('is tabbable and triggers onChange via keyboard', () => {
    const mockFn = jest.fn()
    render(
      <Radio data-testid="radio-testid" checked={false} onChange={mockFn} />
    )
    const component = screen.getByTestId('radio-testid')

    // Tab to focus component
    userEvent.tab()
    expect(component).toHaveFocus()

    // Toggle value with space
    userEvent.type(component, '{space}', { skipClick: true })

    expect(mockFn).toHaveBeenCalledWith(
      expect.objectContaining({
        target: expect.objectContaining({ checked: false }),
      })
    )
  })

  test("isn't focusable when disabled", () => {
    render(<Radio data-testid="radio-testid" disabled />)
    const component = screen.getByTestId('radio-testid')

    userEvent.tab()
    expect(component).not.toHaveFocus()
  })
})

describe('accessibility', () => {
  test('requires label or aria-label to be accessible (case: none)', async () => {
    const { container } = render(<Radio />)
    expect(await axe(container)).not.toHaveNoViolations()
  })

  test('requires label or aria-label to be accessible (case: label)', async () => {
    const { container } = render(<Radio label="Label" />)
    expect(await axe(container)).toHaveNoViolations()
  })

  test('requires label or aria-label to be accessible (case: aria-label)', async () => {
    const { container } = render(<Radio aria-label="Label" />)
    expect(await axe(container)).toHaveNoViolations()
  })

  test('accessible when not checked', async () => {
    const { container } = render(
      <Radio checked={false} onChange={jest.fn} label="Label" />
    )
    expect(await axe(container)).toHaveNoViolations()
  })

  test('accessible when checked', async () => {
    const { container } = render(
      <Radio checked onChange={jest.fn} label="Label" />
    )
    expect(await axe(container)).toHaveNoViolations()
  })
})
