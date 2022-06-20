import { render, screen } from '@testing-library/react'
import cases from 'jest-in-case'
import { axe } from 'jest-axe'

import { Checkbox } from '../Checkbox'

cases(
  'styles:classes',
  ({ params, labelClasses, iconClasses }) => {
    render(
      <Checkbox
        labelProps={{ 'data-testid': 'label-testid' }}
        iconProps={{ 'data-testid': 'icon-testid' }}
        {...params}
      />
    )

    const labelElement = screen.getByTestId('label-testid')
    const iconElement = screen.getByTestId('icon-testid')
    expect(labelElement).toHaveClass(labelClasses)
    expect(iconElement).toHaveClass(iconClasses)
  },
  [
    {
      name: 'default: gray text, icon with transparent bg and gray border',
      params: {},
      labelClasses: 'text-gray-600',
      iconClasses: 'bg-transparent border-2 border-gray-400',
    },
    {
      name: 'checked state: gray text, icon with primary bg and border the same color',
      params: { checked: true, onChange: jest.fn },
      labelClasses: 'text-gray-600',
      iconClasses: 'bg-primary-500 border-2 border-primary-500',
    },
    {
      name: 'indeterminate state: gray text, icon with primary bg and border the same color',
      params: { indeterminate: true, onChange: jest.fn },
      labelClasses: 'text-gray-600',
      iconClasses: 'bg-primary-500 border-2 border-primary-500',
    },
    {
      name: 'disabled default: opacity 40%, icon with primary light bg',
      params: { disabled: true },
      labelClasses: 'text-opacity-40',
      iconClasses: 'bg-primary-100 border-2 border-primary-100',
    },
    {
      name: 'dark default: gray text, icon with transparent bg and gray border',
      params: {},
      labelClasses: 'dark:text-gray-200',
      iconClasses: 'bg-transparent border-2 dark:border-gray-200',
    },
    {
      name: 'dark checked state: gray text, icon with primary bg and border the same color',
      params: { checked: true, onChange: jest.fn },
      labelClasses: 'dark:text-gray-200',
      iconClasses:
        'border-2 dark:bg-primary-400 dark:border-primary-400 dark:text-gray-900',
    },
    {
      name: 'dark indeterminate state: gray text, icon with primary bg and border the same color',
      params: { indeterminate: true, onChange: jest.fn },
      labelClasses: 'dark:text-gray-200',
      iconClasses:
        'border-2 dark:bg-primary-400 dark:border-primary-400 dark:text-gray-900',
    },
    {
      name: 'dark disabled default: opacity 40%, icon with primary light bg',
      params: { disabled: true },
      labelClasses: 'text-opacity-40',
      iconClasses:
        'border-2 dark:bg-primary-800 dark:border-primary-800 dark:text-gray-900',
    },
  ]
)

describe('accessibility', () => {
  test('component is inaccessible when no label is passed', async () => {
    const { container } = render(<Checkbox data-testid="test-id" />)
    expect(await axe(container)).not.toHaveNoViolations()
  })

  test('component is accesible when aria-label is passed', async () => {
    const { container } = render(
      <Checkbox aria-label="Label" data-testid="test-id" />
    )
    expect(await axe(container)).toHaveNoViolations()
  })

  test("aria checked equals false when isn't checked or indeterminate", async () => {
    const { container } = render(
      <Checkbox data-testid="test-id" label="Label" />
    )
    const input = screen.getByRole('checkbox')
    expect(input).toHaveAttribute('aria-checked', 'false')
    expect(await axe(container)).toHaveNoViolations()
  })

  test('aria checked equals true when is checked', async () => {
    const { container } = render(
      <Checkbox
        checked
        data-testid="test-id"
        onChange={jest.fn}
        label="Label"
      />
    )
    const input = screen.getByTestId('test-id')
    expect(input).toHaveAttribute('aria-checked', 'true')
    expect(await axe(container)).toHaveNoViolations()
  })

  test('aria checked equals mixed when is indeterminate', async () => {
    const { container } = render(
      <Checkbox
        indeterminate
        onChange={jest.fn}
        data-testid="test-id"
        label="Label"
      />
    )
    const input = screen.getByTestId('test-id')
    expect(input).toHaveAttribute('aria-checked', 'mixed')
    expect(await axe(container)).toHaveNoViolations()
  })
})
