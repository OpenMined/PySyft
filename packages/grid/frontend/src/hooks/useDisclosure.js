import { useState, useCallback } from 'react'

export const useDisclosure = (startOpen = false) => {
  const [isOpen, setOpen] = useState(startOpen)
  const open = useCallback(() => setOpen(true), [])
  const close = useCallback(() => setOpen(false), [])
  return { isOpen, open, close }
}
