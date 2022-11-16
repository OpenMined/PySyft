export function DetailCard({ spacing = 2, children }) {
  return (
    <div
      className={`pl-6 pr-4 pt-2 pb-4 border border-gray-100 rounded-md space-y-${spacing}`}
      style={{
        background:
          'linear-gradient(90deg, rgba(255, 255, 255, 0.8) 0%, rgba(255, 255, 255, 0.5) 100%), #F1F0F4',
      }}
    >
      {children}
    </div>
  )
}
