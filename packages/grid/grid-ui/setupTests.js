import '@testing-library/jest-dom/extend-expect'
import {configure, act} from '@testing-library/react'

process.env.DEBUG_PRINT_LIMIT = 1000

configure({defaultHidden: true})

afterEach(() => {
  if (jest.isMockFunction(setTimeout)) {
    act(() => jest.runOnlyPendingTimers())
    jest.useRealTimers()
  }
})
