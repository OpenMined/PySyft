describe('PyGrid Domain integration test', () => {
  it('should be able to login the default user', () => {
    cy.visit('http://localhost:9082/login')
    cy.findByText('This domain is running')
  })
})
