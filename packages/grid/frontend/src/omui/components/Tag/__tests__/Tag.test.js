import { render, screen, within, getNodeText, getByTestId } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import cases from 'jest-in-case'

import { Tag } from '../Tag'

const RandomIcon = props => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 496 512" data-testid="test-icon" {...props}>
    <path
      fill="currentColor"
      d="M248 8C111 8 0 119 0 256s111 248 248 248 248-111 248-248S385 8 248 8zm80 168c17.7 0 32 14.3 32 32s-14.3 32-32 32-32-14.3-32-32 14.3-32 32-32zm-160 0c17.7 0 32 14.3 32 32s-14.3 32-32 32-32-14.3-32-32 14.3-32 32-32zm194.8 170.2C334.3 380.4 292.5 400 248 400s-86.3-19.6-114.8-53.8c-13.6-16.3 11-36.7 24.6-20.5 22.4 26.9 55.2 42.2 90.2 42.2s67.8-15.4 90.2-42.2c13.4-16.2 38.1 4.2 24.6 20.5z"
    />
  </svg>
)

describe('Tag', () => {
  cases(
    'styles:classes',
    ({ params, result }) => {
      render(
        <Tag data-testid="test-id" {...params}>
          Omui
        </Tag>
      )

      const tagElement = screen.getByTestId('test-id')
      expect(tagElement).toHaveClass(result)
    },
    [
      {
        name: 'default should be square and have primary colors',
        params: {},
        result: 'bg-primary-100 text-primary-600 rounded-sm',
      },
      {
        name: 'gray tag has different classes',
        params: { variant: 'gray' },
        result: 'bg-gray-100 text-gray-600 hover:bg-gray-800 hover:text-primary-200',
      },
      {
        name: 'can be disabled',
        params: { disabled: true },
        result: 'opacity-50 pointer-events-none',
      },
      {
        name: 'the cursor is pointer when the tag is clickable',
        params: { onClick: jest.fn },
        result: 'cursor-pointer',
      },
      {
        name: 'the tag is round when tagType is round',
        params: { tagType: 'round' },
        result: 'rounded-full',
      },
    ]
  ),
    cases(
      'styles:typography',
      ({ params, result }) => {
        render(
          <Tag data-testid="test-id" {...params}>
            Omui
          </Tag>
        )

        const tagElement = screen.getByTestId('test-id')
        const textElement = tagElement.querySelector('span')
        expect(textElement).toHaveClass(result)
      },
      [
        {
          name: 'default text size should be md',
          params: {},
          result: 'text-md',
        },
        {
          name: 'size=md produces a tag with text size equal to text-md',
          params: { size: 'md' },
          result: 'text-md',
        },

        {
          name: 'size=lg produces a tag with text size equal to text-lg',
          params: { size: 'lg' },
          result: 'text-lg',
        },
        {
          name: 'size=sm produces a tag with text size equal to text-sm',
          params: { size: 'sm' },
          result: 'text-sm',
        },
      ]
    ),
    cases(
      'render:icons',
      ({ params }) => {
        const testString = 'OMui'

        render(
          <Tag data-testid="test-id" {...params}>
            {testString}
          </Tag>
        )

        const tagElement = screen.getByTestId('test-id')

        const randomIconElement = within(tagElement).getByTestId('test-icon')
        expect(randomIconElement).toBeTruthy()

        const tagSpanElements = tagElement.querySelectorAll('span')

        if (params.iconSide === 'left') {
          expect(getByTestId(tagSpanElements[0], 'test-icon')).toBeTruthy()
          expect(getNodeText(tagSpanElements[0])).not.toMatch(testString)
          expect(getNodeText(tagSpanElements[1])).toMatch(testString)
        } else if (params.iconSide === 'right') {
          expect(getNodeText(tagSpanElements[0])).toMatch(testString)
          expect(getNodeText(tagSpanElements[1])).not.toMatch(testString)
          expect(getByTestId(tagSpanElements[1], 'test-icon')).toBeTruthy()
        }
      },
      [
        {
          name: 'Should render icon on left side',
          params: { icon: RandomIcon, iconSide: 'left' },
        },
        {
          name: 'Should render icon on right side',
          params: { icon: RandomIcon, iconSide: 'right' },
        },
      ]
    )

  describe('props', () => {
    test('default tag is a span element', () => {
      render(<Tag data-testid="test-id">Omui</Tag>)

      const tagElement = screen.getByTestId('test-id')
      expect(tagElement.tagName).toBe('SPAN')
    })

    test('the tag is a button when onClick is set', () => {
      render(
        <Tag data-testid="test-id" onClick={jest.fn}>
          Omui
        </Tag>
      )

      const tagElement = screen.getByTestId('test-id')
      expect(tagElement.tagName).toBe('BUTTON')
    })

    test('clicking on a tag works', () => {
      const mockFn = jest.fn()
      render(
        <Tag data-testid="test-id" onClick={mockFn}>
          Omui
        </Tag>
      )

      expect(mockFn).toHaveBeenCalledTimes(0)
      const tagElement = screen.getByTestId('test-id')
      userEvent.click(tagElement)
      expect(mockFn).toHaveBeenCalledTimes(1)
    })
  })
})
