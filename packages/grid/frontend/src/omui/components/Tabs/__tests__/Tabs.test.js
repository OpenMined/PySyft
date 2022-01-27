import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import cases from 'jest-in-case'
import { axe } from 'jest-axe'

import { Tabs } from '../Tabs'

const tabsList = [
  { id: 1, title: 'Tab Name' },
  { id: 2, title: 'Tab Name' },
  { id: 3, title: 'Tab Name', disabled: true },
  { id: 4, title: 'Tab Name' },
  { id: 5, title: 'Tab Name' },
]

cases(
  'styles:classes',
  ({ params, listClasses, tabClasses }) => {
    render(<Tabs tabsList={tabsList} onChange={jest.fn} {...params} />)

    if (listClasses) {
      const tabListElement = screen.getByRole('tablist')
      expect(tabListElement).toHaveClass(listClasses)
    }
    if (tabClasses) {
      const tabElement = screen.getAllByRole('tab')[params.disabled ? 2 : 1]
      expect(tabElement).toHaveClass(tabClasses)
    }
  },
  [
    {
      name: 'default: size md, default state and outlined variant',
      params: {},
      listClasses: 'border-b-2 flex w-full border-primary-500',
      tabClasses: 'py-2.5 px-4 text-md',
    },
    {
      name: 'size: size should be sm',
      params: { size: 'sm' },
      listClasses: 'border-b-2 flex w-full border-primary-500',
      tabClasses: 'py-2 px-2.5 text-sm',
    },
    {
      name: 'size: size should be lg',
      params: { size: 'lg' },
      listClasses: 'border-b-2 flex w-full border-primary-500',
      tabClasses: 'py-4 px-8 text-lg',
    },
    {
      name: 'size: size should be xl',
      params: { size: 'xl' },
      listClasses: 'border-b-2 flex w-full border-primary-500',
      tabClasses: 'py-4.5 px-8 text-xl',
    },
    {
      name: 'custom-state/disabled: text opacity',
      params: { disabled: true },
      tabClasses: 'text-opacity-40 pointer-events-none',
    },
    {
      name: 'variant: outline default state',
      params: { variant: 'outline' },
      listClasses: 'border-primary-500',
      tabClasses: 'border-b-2',
    },
    {
      name: 'variant: outline active state',
      params: { active: 2, variant: 'outline' },
      tabClasses: 'bg-white border-l-2 border-r-2 border-t-2 rounded-t',
    },
    {
      name: 'variant: underline default state',
      params: { variant: 'underline' },
      listClasses: 'border-gray-200',
      tabClasses: 'border-gray-200',
    },
    {
      name: 'variant: underline active state',
      params: { active: 2, variant: 'underline' },
      tabClasses: 'border-primary-500',
    },
  ]
)

describe('mouse navigation', () => {
  test('options are selectable via mouse clicks', () => {
    const mockFn = jest.fn()
    render(<Tabs onChange={mockFn} tabsList={tabsList} />)
    const secondTab = screen.getAllByRole('tab')[1]

    userEvent.click(secondTab)
    expect(mockFn).toHaveBeenCalledWith(tabsList[1].id)
  })

  test('tab is not clickable when disabled', () => {
    const mockFn = jest.fn()
    render(<Tabs active={1} onChange={mockFn} tabsList={tabsList} />)
    const disabledTab = screen.getAllByRole('tab')[2]

    userEvent.click(disabledTab)
    expect(mockFn).not.toHaveBeenCalled()
  })
})

describe('keyboard navigation', () => {
  test('active tab is tabbable and can be navigated with arrow keys', () => {
    const mockFn = jest.fn()
    const { rerender } = render(<Tabs active={1} onChange={mockFn} tabsList={tabsList} />)
    const [first, second] = screen.getAllByRole('tab')

    // Tab to focus active tab
    userEvent.tab()
    expect(first).toHaveFocus()

    userEvent.type(first, '{arrowright}', { skipClick: true })
    expect(mockFn).toHaveBeenCalledWith(tabsList[1].id)
    rerender(<Tabs active={tabsList[1].id} onChange={mockFn} tabsList={tabsList} />)
    expect(second).toHaveFocus()
  })

  test('should focus only the active tab', () => {
    const mockFn = jest.fn()
    const { rerender } = render(<Tabs active={4} onChange={mockFn} tabsList={tabsList} />)
    const [_first, _second, _third, fourth, fifth] = screen.getAllByRole('tab')

    userEvent.tab()
    expect(fourth).toHaveFocus()

    // Tab should not focus any other tabs
    // the navigation is active using arrow keys
    userEvent.tab()
    expect(fifth).not.toHaveFocus()
  })

  test('disabled tabs are not tabbable and focusable', () => {
    const mockFn = jest.fn()
    const { rerender } = render(<Tabs active={2} onChange={mockFn} tabsList={tabsList} />)
    const [_first, second, _third, fourth] = screen.getAllByRole('tab')

    // Tab to focus component
    userEvent.tab()
    expect(second).toHaveFocus()

    // Should skip tabsList[2] (id equals 3) because it is disabled.
    userEvent.type(second, '{arrowright}', { skipClick: true })
    expect(mockFn).toHaveBeenCalledWith(tabsList[3].id)
    rerender(<Tabs active={tabsList[3].id} onChange={mockFn} tabsList={tabsList} />)
    expect(fourth).toHaveFocus()
  })
})

describe('accessibility', () => {
  test('tab list is accessible and aria orientation as horizontal', async () => {
    const mockFn = jest.fn()
    render(<Tabs active={1} onChange={mockFn} tabsList={tabsList} />)

    const tabList = screen.getByRole('tablist')
    expect(tabList).toHaveAttribute('aria-orientation', 'horizontal')

    expect(await axe(tabList)).toHaveNoViolations()
  })

  test('selected tab is accessible and have aria selected as true', async () => {
    const mockFn = jest.fn()
    render(<Tabs active={1} onChange={mockFn} tabsList={tabsList} />)

    const selectedTab = screen.getAllByRole('tab')[0]
    expect(selectedTab).toHaveAttribute('aria-selected', 'true')

    expect(await axe(selectedTab)).toHaveNoViolations()
  })

  test('not selected tab is accessible and have aria selected as false', async () => {
    const mockFn = jest.fn()
    render(<Tabs active={1} onChange={mockFn} tabsList={tabsList} />)

    const anyTab = screen.getAllByRole('tab')[3]
    expect(anyTab).toHaveAttribute('aria-selected', 'false')

    expect(await axe(anyTab)).toHaveNoViolations()
  })

  test('tab panel is accessible and is labelled by active tab', async () => {
    const mockFn = jest.fn()
    render(<Tabs active={1} onChange={mockFn} tabsList={tabsList} />)

    const tabPanel = screen.getByRole('tabpanel')
    expect(tabPanel).toHaveAttribute('aria-labelledby', 'omui-tab-1')

    expect(await axe(tabPanel)).toHaveNoViolations()
  })
})
