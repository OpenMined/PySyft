describe('PyGrid Domain integration test', () => {
  it('should be able to login the default user', () => {
    cy.visit('/login')
    cy.findByText(/This domain is running/)
    cy.findByPlaceholderText(/abc@university.edu/).type('info@openmined.org')
    cy.findByPlaceholderText('···········').type('changethis')
    cy.findByRole('button', {name: /login/i}).click()
    cy.location('pathname').should('equal', '/users')
  })
})
