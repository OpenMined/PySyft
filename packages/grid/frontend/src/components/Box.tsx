interface BoxProps {
  cols?: number | 'full'
}

function Box({ cols = 'full', ...props }: BoxProps) {
  return <div {...props} className={cols && `col-span-${cols}`} />
}

export { Box }
