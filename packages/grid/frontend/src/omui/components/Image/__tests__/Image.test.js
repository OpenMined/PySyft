import { render, screen } from '@testing-library/react'
import { axe } from 'jest-axe'
import cases from 'jest-in-case'

import { Image } from '../Image'

cases(
  'styles:classes',
  ({ params, result }) => {
    render(<Image containerProps={{ 'data-testid': 'test-id' }} {...params} />)

    const imageElement = screen.getByTestId('test-id')
    expect(imageElement).toHaveClass(result)
  },
  [
    {
      name: 'Should not have aspect ratio reversed',
      params: {},
      result: 'aspect-w-16 aspect-h-9',
    },
    {
      name: 'Should reverse aspect ratio for portrait images',
      params: { orientation: 'portrait' },
      result: 'aspect-w-9 aspect-h-16',
    },
  ]
)

describe('accessibility', () => {
  test('Image has role=img when alt text is set and no other axe violations', async () => {
    const { container, getByRole } = render(
      <Image containerProps={{ 'data-testid': 'test-id' }} alt="Alternative text" />
    )
    const img = getByRole('img')
    expect(img).toBeInTheDocument()
    const results = await axe(container)
    expect(results).toHaveNoViolations()
  })

  test('Image has role=presentation when no alt text is set and no other axe violations', async () => {
    const { container, getByRole } = render(<Image containerProps={{ 'data-testid': 'test-id' }} />)
    const img = getByRole('presentation')
    expect(img).toBeInTheDocument()
    const results = await axe(container)
    expect(results).toHaveNoViolations()
  })
})
