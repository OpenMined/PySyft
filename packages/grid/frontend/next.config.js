const nodeType = process.env.NODE_TYPE
const domainType = nodeType?.toLowerCase() === 'network' ? 'network' : 'domain'

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
