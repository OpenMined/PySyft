import { render, screen } from '@testing-library/react'
import cases from 'jest-in-case'

import { Input } from '../Input'

const Addon = <i />

cases(
  'Validate addons rendering',
  ({ params, inDocument = [], notInDocument = [] }) => {
    render(<Input data-testid="test-id" {...params} />)

    inDocument.forEach(element => {
      expect(screen.queryByTestId(element)).toBeInTheDocument()
    })
    notInDocument.forEach(element => {
      expect(screen.queryByTestId(element)).not.toBeInTheDocument()
    })
  },
  [
    {
      name: 'Should render only left addon',
      params: { addonLeft: Addon, addonLeftProps: { 'data-testid': 'left-addon-id' } },
      inDocument: ['left-addon-id'],
      notInDocument: ['right-addon-id'],
    },
    {
      name: 'Should render only right addon',
      params: { addonRight: Addon, addonRightProps: { 'data-testid': 'right-addon-id' } },
      inDocument: ['right-addon-id'],
      notInDocument: ['left-addon-id'],
    },
    {
      name: 'Should render both addons',
      params: {
        addonLeft: Addon,
        addonLeftProps: { 'data-testid': 'left-addon-id' },
        addonRight: Addon,
        addonRightProps: { 'data-testid': 'right-addon-id' },
      },
      inDocument: ['right-addon-id', 'left-addon-id'],
    },
  ]
)

cases(
  'Validate aria attributes by prop',
  ({ params, result = [], resultNot = [] }) => {
    render(<Input data-testid="test-id" {...params} />)

    const inputElement = screen.getByTestId('test-id')
    result.forEach(({ attribute, value }) => {
      expect(inputElement).toHaveAttribute(attribute, value)
    })
    resultNot.forEach(({ attribute }) => {
      expect(inputElement).not.toHaveAttribute(attribute)
    })
  },
  [
    {
      name: 'Should only have invalid aria attribute',
      params: { error: true },
      result: [{ attribute: 'aria-invalid', value: 'true' }],
      resultNot: [{ attribute: 'aria-required' }, { attribute: 'aria-readonly' }],
    },
    {
      name: 'Should not have invalid aria attribute',
      params: {},
      resultNot: [{ attribute: 'aria-invalid' }],
    },
    {
      name: 'Should have required aria attribute',
      params: { required: true },
      result: [
        { attribute: 'aria-required', value: 'true' },
        { attribute: 'required', value: '' },
      ],
    },
    {
      name: 'Should have readOnly aria attribute',
      params: { readOnly: true },
      result: [{ attribute: 'aria-readonly', value: 'true' }],
    },
  ]
)
