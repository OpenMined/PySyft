const Module = require('module')
const path = require('path')
const resolveFrom = require('resolve-from')
const node_modules = path.resolve(__dirname, 'node_modules')
const originalRequire = Module.prototype.require

const nodeType = process.env.NODE_TYPE
const domainType = nodeType?.toLowerCase() === 'network' ? 'network' : 'domain'

Module.prototype.require = function (modulePath) {
  // Only redirect resolutions to non-relative and non-absolute modules
  if (
    ['/react/', '/react-dom/', '/react-query/'].some((d) => {
      try {
        return require.resolve(modulePath).includes(d)
      } catch (err) {
        return false
      }
    })
  ) {
    try {
      modulePath = resolveFrom(node_modules, modulePath)
    } catch (err) {
      //
    }
  }

  return originalRequire.call(this, modulePath)
}

module.exports = {
  typescript: {
    ignoreDevErrors: true,
    ignoreBuildErrors: true,
  },
  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },
  webpackDevMiddleware: (config) => {
    config.watchOptions = {
      poll: 1000,
      aggregateTimeout: 300,
    }
    return config
  },
  webpack: (config) => {
    config.resolve = {
      ...config.resolve,
      alias: {
        ...config.resolve.alias,
        react$: resolveFrom(path.resolve('node_modules'), 'react'),
        'react-query$': resolveFrom(
          path.resolve('node_modules'),
          'react-query'
        ),
        'react-dom$': resolveFrom(path.resolve('node_modules'), 'react-dom'),
      },
    }
    return config
  },
  env: {
    NEXT_PUBLIC_VERSION: process.env.VERSION,
    NEXT_PUBLIC_VERSION_HASH: process.env.VERSION_HASH,
    NEXT_PUBLIC_NODE_TYPE: process.env.NODE_TYPE,
  },
  async redirects() {
    return [
      {
        source: '/_network/:path*',
        destination: '/:path*',
        permanent: false,
      },
      {
        source: '/_domain/:path*',
        destination: '/:path*',
        permanent: false,
      },
    ]
  },
  async rewrites() {
    return [
      {
        source: '/:path*',
        destination: `/_${domainType}/:path*`,
      },
    ]
  },
}
