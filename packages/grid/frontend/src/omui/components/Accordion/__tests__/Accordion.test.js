import { render, screen } from '@testing-library/react'
import cases from 'jest-in-case'
import userEvent from '@testing-library/user-event'
import { axe } from 'jest-axe'

import {
  Accordion,
  AccordionButton,
  AccordionItem,
  AccordionPanel,
} from '../Accordion'

const mockItems = [
  {
    button: 'Foundations',
    panel: 'Content for Foundations of Private Computation',
  },
  { button: 'Privacy', panel: 'Our Privacy Opportunity', disabled: true },
  {
    button: 'Federated Learning',
    panel: 'Federated Learning Across Enterprises',
  },
]

const mockComponent = (params) => {
  return (
    <Accordion data-testid="test-id" defaultIndex={2} {...params}>
      {mockItems.map(({ button, panel, disabled }, idx) => (
        <AccordionItem disabled={disabled} key={idx}>
          <AccordionButton>{button}</AccordionButton>
          <AccordionPanel>{panel}</AccordionPanel>
        </AccordionItem>
      ))}
    </Accordion>
  )
}

cases(
  'styles:classes',
  ({ params, index = 0, result }) => {
    render(mockComponent(params))

    const item = screen.getAllByRole(/button/)[index].parentElement
    expect(item).toHaveClass(result)
  },
  [
    {
      name: 'default: default state',
      result: 'border-gray-200 border-t border-b',
    },
    {
      name: 'custom-state/disabled: default size and disabled state',
      index: 1,
      result: 'opacity-40 pointer-events-none',
    },
  ]
)

describe('mouse navigation', () => {
  test('defaultIndex is initially active', () => {
    render(mockComponent({ defaultIndex: 0 }))
    const firstButton = screen.getAllByRole('button')[0]

    expect(firstButton).toHaveAttribute('aria-expanded', 'true')
  })

  test('accordion can be state controlled', () => {
    const mockFn = jest.fn()
    const index = 0
    render(mockComponent({ index, onChange: mockFn }))
    const button = screen.getAllByRole('button')[2]

    userEvent.click(button)
    expect(mockFn).toHaveBeenCalledWith(2)
  })

  test('accordion item can be closed', () => {
    const mockFn = jest.fn()
    const index = 0
    render(mockComponent({ index, onChange: mockFn }))
    const button = screen.getAllByRole('button')[0]

    userEvent.click(button)
    expect(mockFn).toHaveBeenCalledWith(null)
  })

  test('item is not clickable when disabled', () => {
    const mockFn = jest.fn()
    render(mockComponent({ onChange: mockFn }))
    const disabledButton = screen.getAllByRole('button')[1]

    userEvent.click(disabledButton)
    expect(mockFn).not.toHaveBeenCalled()
  })
})

describe('keyboard navigation', () => {
  test('accordion item can be toggled with space or enter', () => {
    render(mockComponent({ defaultIndex: 0 }))
    const button = screen.getAllByRole('button')[0]

    // initially opened
    expect(button).toHaveAttribute('aria-expanded', 'true')

    // Tab to focus component
    userEvent.tab()
    expect(button).toHaveFocus()

    userEvent.type(button, '{space}', { skipClick: true })
    expect(button).toHaveAttribute('aria-expanded', 'false')

    userEvent.type(button, '{enter}', { skipClick: true })
    expect(button).toHaveAttribute('aria-expanded', 'true')
  })

  test('item is not focusable when disabled', () => {
    render(mockComponent())
    const [first, second, third] = screen.getAllByRole('button')

    userEvent.tab()
    expect(first).toHaveFocus()

    // disabled element should be skipped
    userEvent.tab()
    expect(second).not.toHaveFocus()

    expect(third).toHaveFocus()
  })
})

describe('accessibility', () => {
  test('aria attributes are correctly set', async () => {
    const { container } = render(mockComponent({ defaultIndex: 2 }))
    const [first, second, third] = screen.getAllByRole('button')
    const regions = screen.getAllByRole('region')

    expect(first).toHaveAttribute('aria-expanded', 'false')
    expect(second).toHaveAttribute('aria-expanded', 'false')
    expect(third).toHaveAttribute('aria-expanded', 'true')

    expect(await axe(container)).toHaveNoViolations()
  })

  test('only the active region should not be hidden', async () => {
    const { container } = render(mockComponent({ defaultIndex: 0 }))
    const firstButton = screen.getAllByRole('button')[0]
    const firstPanel = container.querySelector('#omui-accordion-panel-0')
    const secondPanel = container.querySelector('#omui-accordion-panel-2')

    expect(firstButton).toHaveAttribute('aria-expanded', 'true')
    expect(firstPanel).toHaveAttribute('aria-hidden', 'false')
    expect(secondPanel).toHaveAttribute('aria-hidden', 'true')
    expect(await axe(container)).toHaveNoViolations()
  })
})
