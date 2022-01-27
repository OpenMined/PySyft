import cn from 'classnames'

export function Overlay({ children }) {
  return (
    <div className={cn('fixed inset-0 w-full h-screen bg-gray-800 bg-opacity-40 z-10')}>
      {children}
    </div>
  )
}
