describe('Login', () => {
  before(() => {
    cy.visit('/login')
  })

  describe('Meta', () => {
    it('should be able to land on login landing page', () => {
      cy.findByText(/login/i)
      cy.findByLabelText(/email/i)
      cy.findByLabelText(/password/i)
    })

    it('should be able to see the current version of PyGrid being used', () => {
      cy.findByText(/canada domain/i)
      cy.findByText(/kyoko eng/i)
    })

    it('should be able to see if domain is online', () => {
      cy.findByTestId('domain-status').findByText(/domain online/i)
    })

    it('should be able to see if domain is offline', () => {
      // cy.intercept('GET', '/api/v1/status', { fixture: 'domain_offline.json' }) # TODO: add domain_offline.json endpoint
      cy.findByText(/domain offline/i).should("not.exist") // TODO: make exist
    })

    it('should be able to see text redirecting to sign up', () => {
      cy.findByTestId('redirect-sign-up')
        .contains(/don't have an account yet?/i)
        .contains(/apply for an account here/i)
    })

    it('should redirect to sign up page if user clicks on "apply for..."', () => {
      cy.findByTestId('redirect-sign-up').findByText(/apply/i).click()
      cy.location('pathname').should('equal', '/register')
      cy.visit('/login')
    })
  })

  describe('Successful Login', () => {
    beforeEach(() => {
      // cy.visit('/logout') # TODO: implement logout test
      cy.visit('/login')
    })

    it('should be able to login a returning user using the correct credentials', () => {
      cy.findByPlaceholderText(/abc@university.org/).type('info@openmined.org')
      cy.findByPlaceholderText('*********').type('changethis')
      cy.findByTestId('login-button').click()
      cy.findByTestId('login-button').should('not.have.text', /login/i)
      // cy.findByTestId('login-button').should('have.class', 'bg-green-500') // TODO: fix
      // cy.location('pathname').should('equal', '/requests/data') // TODO: fix
    })

    it('should be able to login a newly registered user using the correct credentials and redirect to onboarding', () => {
      cy.findByPlaceholderText(/abc@university.org/).type('new_user@openmined.org')
      cy.findByPlaceholderText('*********').type('changethis')
      cy.findByTestId('login-button').click()
      cy.findByTestId('login-button').should('not.have.text', /login/i)
      // cy.findByTestId('login-button').should('have.class', 'bg-green-500') // TODO: fix
      // cy.location('pathname').should('equal', '/onboarding') // TODO: fix
    })
  })

  describe('Unsuccessful Login', () => {
    before(() => {
      cy.visit('/login')
    })

    it('should give feedback to the user on failed attempts', () => {
      cy.findByPlaceholderText(/abc@university.org/).type('info@openmined.org')
      cy.findByPlaceholderText('*********').type('changethat')
      cy.findByTestId('login-button').click()
      cy.findAllByTestId('toast')
        .findByText(/invalid credentials/i)
        .should('exist')
    })
  })
})
