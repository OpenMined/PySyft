import cn from 'classnames'

const Grid = props => (
  <div
    {...props}
    className={cn('grid grid-cols-12 gap-2 px-10 min-h-full w-full', props.className)}
  />
)

export { Grid }
