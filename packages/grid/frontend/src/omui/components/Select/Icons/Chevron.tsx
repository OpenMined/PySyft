import React from 'react'
import type { SVGProps } from 'react'

const Chevron = (props: SVGProps<SVGSVGElement>) => {
  return (
    <svg
      width="12"
      height="8"
      viewBox="0 0 12 8"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path
        d="M1.5 1.75L6 6.25L10.5 1.75"
        stroke="currentColor"
        strokeWidth="2.25"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export { Chevron }
