module.exports = (on, config) => {
  Object.assign(config, {integrationFolder: 'cypress/e2e'})
  return config
}
