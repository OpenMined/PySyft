import React from 'react'
import type { SVGProps } from 'react'

const Indeterminate = (props: SVGProps<SVGSVGElement>) => {
  return (
    <svg
      width="8"
      height="2"
      viewBox="0 0 8 2"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M0 1H8" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  )
}

export { Indeterminate }
