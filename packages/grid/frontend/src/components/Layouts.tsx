import { Page } from '@/components/Page'
import { Grid } from '@/components/Grid'
import { SidebarNav } from '@/components/SidebarNav'

export function Base({ children }) {
  return (
    <div
      className="omui-layout"
      style={{
        background:
          'linear-gradient(90deg, rgba(0, 0, 0, 0.75) 0%, rgba(0, 0, 0, 0.1) 100%), #2E2B3B',
      }}
    >
      <SidebarNav />
      <main className="bg-white rounded-tl-2xl relative">
        <Page>
          <Grid>{children}</Grid>
        </Page>
        <div id="omui-modal-portal" />
      </main>
    </div>
  )
}

export function SingleCenter({ children }) {
  return (
    <Base>
      <div className="col-span-10 col-start-2 grid grid-cols-10">
        {children}
      </div>
    </Base>
  )
}
