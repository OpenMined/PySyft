import '@testing-library/cypress/add-commands'
import { configure } from '@testing-library/cypress'

configure({ testIdAttribute: 'data-cy' })
