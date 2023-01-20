import cn from 'classnames'

export function BorderedBox(props) {
  return (
    <div
      {...props}
      className={cn('border border-gray-100 rounded p-6', props.className)}
    />
  )
}
