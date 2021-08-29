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
  }
}
