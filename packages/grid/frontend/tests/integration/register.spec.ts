describe('Register', () => {
  before(() => {
    cy.visit('/register')
  })

  describe('Meta', () => {
    it('should show domain information', () => {
      cy.findByTestId('meta-domain-name').should('not.be.empty')
      cy.findByTestId('meta-domain-tags').should('be.visible')
      cy.findByTestId('meta-domain-description').should('be.visible')
      cy.findByTestId('meta-domain-number-datasets').should('be.visible')
      cy.findByTestId('meta-domain-owner').should('be.visible')
      cy.findByTestId('meta-domain-institution').should('be.visible')
      cy.findByTestId('meta-domain-joined_networks').should('be.visible')
      cy.findByTestId('meta-domain-support_email').should('be.visible')
    })
  })

  describe('Register without agreement', () => {
    before(() => {
      cy.visit('/register')
    })

    it('Registers a new user', () => {
      cy.findByPlaceholderText(/jane doe/i).type('Jane Doe')
      cy.findByPlaceholderText(/abc university/i).type('OpenMined')
      cy.findByPlaceholderText(/abc@university.org/i).type('jane.doe@openmined.org')
      cy.findByLabelText('Password *').type('changethis')
      cy.findByLabelText('Confirm Password *').type('changethis')
      cy.findByPlaceholderText(/this can help a domain owner vett your application/i).type(
        'OpenMined'
      )
      cy.findByTestId('register-button').click()
      cy.location('pathname').should('equal', '/login')
    })
  })
})
