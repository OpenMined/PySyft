module.exports = {
  root: true,
  plugins: ['eslint-plugin-cypress'],
  extends: ['plugin:cypress/recommended'],
  env: { 'cypress/globals': true },
}
