import { render, screen } from '@testing-library/react'
import cases from 'jest-in-case'
import userEvent from '@testing-library/user-event'
import { axe } from 'jest-axe'

import { FormControl } from '../FormControl'

const Children = props => (
  <>
    <input id={props.id} />
    {JSON.stringify(props)}
  </>
)

const mockComponent = params => {
  return (
    <FormControl id="mock-id" {...params}>
      <Children />
    </FormControl>
  )
}

describe('Form Control', () => {
  cases(
    'styles:classes',
    ({ params, textClasses }) => {
      render(mockComponent(params))

      const labelItem = screen.getByText(params.label)
      const helperTextItem = screen.getByText(params.helperText)

      expect(labelItem).toHaveClass(textClasses)
      expect(helperTextItem).toHaveClass(textClasses)
    },
    [
      {
        name: 'default: default state',
        params: { label: 'Label', helperText: 'Hint text' },
        textClasses: 'text-gray-500 dark:text-gray-200',
      },
      {
        name: 'custom-state/disabled: disabled state',
        params: { label: 'Label', helperText: 'Hint text', disabled: true },
        textClasses: 'opacity-50 pointer-events-none',
      },
      {
        name: 'custom-state/error: error state',
        params: { label: 'Label', helperText: 'Hint text', error: true },
        textClasses: 'text-error-600 dark:text-error-200 fill-error-200 dark:fill-error-200',
      },
    ]
  )

  cases(
    'some props drill',
    ({ params, result }) => {
      render(mockComponent(params))

      const propsAsStringify = screen.findByText(result)

      expect(propsAsStringify).toBeDefined()
    },
    [
      {
        name: 'default: should not drill label and helper text',
        params: { label: 'Label', helperText: 'Hint text' },
        result: JSON.stringify({}),
      },
      {
        name: 'disabled: should drill disabled prop',
        params: { label: 'Label', disabled: true },
        result: JSON.stringify({ disabled: true }),
      },
      {
        name: 'error: should drill error prop',
        params: { label: 'Label', error: true },
        result: JSON.stringify({ error: true }),
      },
      {
        name: 'required: should drill required prop',
        params: { label: 'Label', required: true },
        result: JSON.stringify({ required: true }),
      },
      {
        name: 'custom-state: should drill id prop',
        params: { label: 'Label', id: 'custom-id' },
        result: JSON.stringify({ id: 'custom-id' }),
      },
      {
        name: 'all props: should drill all form props',
        params: {
          label: 'Label',
          id: 'custom-id',
          disabled: true,
          error: true,
          required: true,
        },
        result: JSON.stringify({ id: 'custom-id', disabled: true, error: true, required: true }),
      },
    ]
  )

  describe('accessibility', () => {
    test('contains axe violations when it does not have a label', async () => {
      const { container } = render(mockComponent())

      expect(await axe(container)).not.toHaveNoViolations()
    })

    test('contains no axe violations when it does have a label', async () => {
      const { container } = render(mockComponent({ label: 'Label' }))

      expect(await axe(container)).toHaveNoViolations()
    })

    test('contains no axe violations when it does have label and helperText', async () => {
      const { container } = render(mockComponent({ label: 'Label', helperText: 'Hint text' }))

      expect(await axe(container)).toHaveNoViolations()
    })
  })
})
