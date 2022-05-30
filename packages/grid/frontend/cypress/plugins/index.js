module.exports = (on, config) => {
  Object.assign(config, { integrationFolder: 'cypress/e2e' })
  config.baseUrl = process.env.NEXT_PUBLIC_HOST
  return config
}
