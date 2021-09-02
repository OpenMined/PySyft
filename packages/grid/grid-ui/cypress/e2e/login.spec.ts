describe('PyGrid UI', () => {
  it('should be able to login the default user', () => {
    cy.visit('/')
    cy.findByLabelText(/email/i).type('info@openmined.org')
    cy.findByLabelText(/password/i).type('changethis')
    cy.findByRole('button', {name: /login/i}).click()
  })
})
