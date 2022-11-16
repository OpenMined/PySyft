import { render, screen } from '@testing-library/react'
import cases from 'jest-in-case'

import { Text } from '../Text'

cases(
  'Validate text classes',
  ({ params, result }) => {
    render(
      <Text data-testid="test-id" {...params}>
        Omui
      </Text>
    )

    const textElement = screen.getByTestId('test-id')
    expect(textElement).toHaveClass(result)
  },
  [
    {
      name: 'Should be default with roboto, normal font and text md',
      params: {},
      result: 'font-roboto font-normal text-md',
    },
    {
      name: 'Should have bold font',
      params: { bold: true },
      result: 'font-roboto font-bold text-md',
    },
    {
      name: 'Should adjust font weight, family and size when size is greater than XL',
      params: { size: 'xl' },
      result: 'font-rubik font-medium text-xl',
    },
    {
      name: 'Should be underline and bold',
      params: { underline: true, bold: true },
      result: 'font-roboto font-bold text-md underline',
    },
    {
      name: 'Should be mono font faced, transform uppercase and ignore underline',
      params: { underline: true, mono: true },
      result: 'font-firacode font-normal text-md uppercase',
    },
    {
      name: 'Should have font weight black when size is greater than XL and bold',
      params: { size: '3xl', bold: true },
      result: 'font-rubik font-black text-3xl',
    },
    {
      name: 'Should not have font weight black when mono font faced',
      params: { size: '3xl', mono: true },
      result: 'font-firacode font-medium text-3xl',
    },
    {
      name: 'Should have custom rules for mono font faced',
      params: { size: '5xl', mono: true, bold: true },
      result: 'font-firacode font-bold text-5xl-mono',
    },
    {
      name: 'Should have custom rules for uppercase',
      params: { size: '5xl', uppercase: true },
      result: 'font-rubik font-medium text-5xl-upper',
    },
    {
      name: 'Should not have custom rules when rules conflicts',
      params: { size: '6xl', uppercase: true, mono: true },
      result: 'font-firacode font-medium text-6xl uppercase',
    },
  ]
)
