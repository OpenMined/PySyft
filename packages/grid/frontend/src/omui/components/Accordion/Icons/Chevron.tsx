import React from 'react'
import type { SVGProps } from 'react'

const Chevron = (props: SVGProps<SVGSVGElement>) => {
  return (
    <svg
      width="19"
      height="12"
      viewBox="0 0 19 12"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      {...props}
    >
      <path
        d="M8.83594 10.8828C9.1875 11.2344 9.77344 11.2344 10.125 10.8828L17.7422 3.30469C18.0938 2.91406 18.0938 2.32812 17.7422 1.97656L16.8438 1.07813C16.4922 0.726563 15.9062 0.726563 15.5156 1.07813L9.5 7.09375L3.44531 1.07813C3.05469 0.726563 2.46875 0.726563 2.11719 1.07813L1.21875 1.97656C0.867188 2.32813 0.867188 2.91406 1.21875 3.30469L8.83594 10.8828Z"
        fill="currentColor"
      />
    </svg>
  )
}

export { Chevron }
