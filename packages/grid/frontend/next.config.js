const nodeType = process.env.NODE_TYPE
const domainType = nodeType?.toLowerCase() === 'network' ? 'network' : 'domain'

module.exports = {
  typescript: {
    ignoreDevErrors: true,
    ignoreBuildErrors: true
  },
  webpackDevMiddleware: config => {
    config.watchOptions = {
      poll: 1000,
      aggregateTimeout: 300
    }
    return config
  },
  async redirects() {
    return [
      {
        source: '/_network/:path*',
        destination: '/:path*',
        permanent: false
      },
      {
        source: '/_domain/:path*',
        destination: '/:path*',
        permanent: false
      }
    ]
  },
  async rewrites() {
    return [
      {
        source: '/:path*',
        destination: `/_${domainType}/:path*`
      }
    ]
  }
}
