import React from 'react'

export default function NetworkIndex() {
  return (
    <article className="bg-white w-full h-full min-h-screen min-w-screen flex flex-col flex-grow justify-center items-center gap-6">
      <img src="/assets/small-logo.png" alt="PyGrid logo" />
      <header className="text-center px-6">
        <h1 className="text-2xl md:text-5xl font-bold mt-6">
          Network is running.
        </h1>
        <p className="md:text-lg mt-8">
          Congratulations! Your PyGrid Network is up and running.
        </p>
        <p className="md:text-lg mt-1">
          Please see the{' '}
          <a
            className="text-primary-500 hover:text-primary-600 hover:underline"
            href="https://courses.openmined.org/courses/introduction-to-remote-data-science"
          >
            Introduction to Remote Data Science Course
          </a>{' '}
          to see how to use it!
        </p>
        <p className="md:text-lg mt-1">
          If you would like your network to be registered in the official
          "syft.networks" registry, please create an issue in the{' '}
          <a
            className="text-primary-500 hover:text-primary-600 hover:underline"
            href="https://github.com/OpenMined/NetworkRegistry"
          >
            OpenMined/NetworkRegistry
          </a>{' '}
          repository.
        </p>
        <p style={{ display: 'none' }}>This is a PyGrid Network node.</p>
      </header>
    </article>
  )
}
