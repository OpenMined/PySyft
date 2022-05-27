import { Spinner } from '@/components'

export function SpinnerWithText({ children }) {
  return (
    <div className="flex space-x-4">
      <p>{children}</p>
      <Spinner className="w-6 h-6 text-gray-500" />
    </div>
  )
}
