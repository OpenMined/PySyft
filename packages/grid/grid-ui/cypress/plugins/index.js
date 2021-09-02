module.exports = (on, config) => {
  config.baseUrl = 'http://localhost:80'
  Object.assign(config, {integrationFolder: 'cypress/e2e'})
  return config
}
