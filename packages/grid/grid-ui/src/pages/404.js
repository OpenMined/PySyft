import React from 'react'
import Link from 'next/link'

const NoMatch = () => {
  return (
    <article className="flex justify-center items-center">
      <div className="text-center justify-items-center grid grid-flow-row auto-rows-max space-y-4">
        <header>
          <h1>Ooops!</h1>
          <p className="subtitle">This page is unknown or does not exist</p>
        </header>
        <Link href="/">
          <button className="w-full btn">Back to Dashboard</button>
        </Link>
      </div>
    </article>
  )
}

export default NoMatch
