module.exports = {
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.js'],
  testPathIgnorePatterns: ['<rootDir>/.next/', '<rootDir>/node_modules/', '<rootDir>/scripts/'],
  setupFilesAfterEnv: ['<rootDir>/setupTests.js'],
  bail: 1,
  collectCoverage: true,
  collectCoverageFrom: ['components/**/*.js', 'pages/**/*.js'],
  coverageReporters: ['lcov', 'text'],
  resetMocks: true,
  moduleNameMapper: {
    '@/(.*)': ['<rootDir>/src/$1'],
  },
}
