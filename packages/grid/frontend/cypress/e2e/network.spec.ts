describe('PyGrid Network integration test', () => {
  it('should be able to login the default user', () => {
    cy.visit('http://localhost:9083/login')
    cy.findByText('type: network')
  })
})
