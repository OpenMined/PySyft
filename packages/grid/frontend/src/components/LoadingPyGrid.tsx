import { Dialog } from '@headlessui/react'

export function LoadingPyGrid() {
  return (
    <Dialog open onClose={() => true} className="fixed inset-0 z-10 overflow-y-auto">
      <Dialog.Overlay className="fixed inset-0 z-10" />
      <div className="flex items-center justify-center min-h-screen mx-auto bg-white min-w-screen transition-all">
        <img
          alt="PyGrid logo"
          src="/assets/logo.png"
          className="z-30 w-24 h-24 opacity-100 animate-pulse"
        />
        <h2 className="animate-pulse">Loading PyGrid...</h2>
      </div>
      <button className="w-px sr-only" onClick={() => false} />
    </Dialog>
  )
}
