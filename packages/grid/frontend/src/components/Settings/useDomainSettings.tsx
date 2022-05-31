import { createContext, useContext } from 'react'

const DomainSettingsContext = createContext({ settings: null })

const useDomainSettings = () => useContext(DomainSettingsContext)

const DomainSettingsProvider = ({ value, children }) => (
  <DomainSettingsContext.Provider value={value}>
    {children}
  </DomainSettingsContext.Provider>
)

export { useDomainSettings, DomainSettingsProvider }
