import React from 'react'
import Head from 'next/head'

export default function Status() {
  return (
    <>
      <Head>
        <title>{process.env.NEXT_PUBLIC_NODE_TYPE}</title>
      </Head>
      <article className="pt-4 w-full flex justify-center flex-col items-center">
        <div className="center">
          <img
            alt="PyGrid logo"
            src="/assets/logo.png"
            width={200}
            height={200}
          />
        </div>
        <div className="my-8">
          <p className="capitalize">
            Type <strong>{process.env.NEXT_PUBLIC_NODE_TYPE}</strong>
          </p>
          <p>
            Version <strong>{process.env.NEXT_PUBLIC_VERSION}</strong>
          </p>
          <p>
            Version (hash){' '}
            <strong>{process.env.NEXT_PUBLIC_VERSION_HASH}</strong>
          </p>
        </div>
      </article>
    </>
  )
}
