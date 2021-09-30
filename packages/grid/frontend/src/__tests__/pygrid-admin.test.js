import React from 'react'
import {render, screen} from '@testing-library/react'
import {AppProviders} from '@/context'
import Homepage from '@/pages/index'
import {useRouter} from 'next/router'

jest.mock('next/router', () => ({
  useRouter: jest.fn()
}))

const replace = jest.fn()

describe('App', () => {
  beforeEach(() => {
    useRouter.mockImplementation(() => ({
      replace
    }))
  })

  it('home redirects to /login if user is not authenticated', async () => {
    render(<Homepage />, {wrapper: AppProviders})
    screen.debug()
    expect(replace).toHaveBeenCalledWith('/login')
  })
})
