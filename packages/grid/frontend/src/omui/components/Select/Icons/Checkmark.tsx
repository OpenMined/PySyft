import React from 'react'
import type { SVGProps } from 'react'

const Checkmark = (props: SVGProps<SVGSVGElement>) => {
  return (
    <svg
      width="12"
      height="10"
      viewBox="0 0 12 10"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path
        d="M11 0.999512L4 8.99951L1 5.99951"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export { Checkmark }
