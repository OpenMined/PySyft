import React from 'react'
import type { SVGProps } from 'react'

const Checkmark = (props: SVGProps<SVGSVGElement>) => {
  return (
    <svg
      width="11"
      height="9"
      viewBox="0 0 11 9"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path d="M10 1.5L4 7.5L1 4.5" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  )
}

export { Checkmark }
